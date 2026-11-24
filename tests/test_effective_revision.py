"""Critic claims must point at real report sentences; revisions must change."""

import json
import unittest

from agents import AnalysisFailedError, CriticAgent, SynthesisAgent
from agents.claims import (
    drop_sentences,
    match_claim,
    report_sentences,
    resolve_claims,
)
from models import ContractValidationError
from services import LLMResult
from workflows import ReviewedSynthesisWorkflow
from test_critic_agent import valid_digest, valid_profile
from test_synthesis_agent import assessed_signal, valid_response


class SequenceClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def complete_json(self, messages, **kwargs):
        self.messages.append(messages)
        return LLMResult(content=next(self.responses), model="test", latency_ms=1)


def critic_json(claims, scores=(90, 90, 90), feedback=None):
    return json.dumps(
        {
            "groundedness_score": scores[0],
            "profile_relevance_score": scores[1],
            "completeness_score": scores[2],
            "unsupported_claims": claims,
            "feedback": feedback or [],
        }
    )


class ClaimMatchingTests(unittest.TestCase):
    def setUp(self):
        self.digest = {
            "executive_summary": (
                "Three items target Python developers. "
                "MASSys benchmarks multi-agent serving substrates."
            ),
            "recommendations": [
                {"text": "Evaluate MASSys for your agent workflows.", "evidence_ids": []}
            ],
        }
        self.sentences = report_sentences(self.digest)

    def test_report_is_split_into_sentences(self):
        self.assertEqual(
            self.sentences,
            [
                "Three items target Python developers.",
                "MASSys benchmarks multi-agent serving substrates.",
                "Evaluate MASSys for your agent workflows.",
            ],
        )

    def test_exact_and_paraphrased_claims_resolve_to_report_sentence(self):
        self.assertEqual(
            match_claim('"Three items target Python developers"', self.sentences),
            "Three items target Python developers.",
        )
        self.assertEqual(
            match_claim("Three items target Python developer", self.sentences),
            "Three items target Python developers.",
        )

    def test_text_not_in_report_is_not_matched(self):
        resolved, not_in_report = resolve_claims(
            ["The developer's interest in AI agents is stated in the profile."],
            self.sentences,
        )
        self.assertEqual(resolved, [])
        self.assertEqual(len(not_in_report), 1)


class CriticClaimResolutionTests(unittest.TestCase):
    def test_claim_outside_report_becomes_visible_feedback_not_a_failure(self):
        client = SequenceClient(
            [critic_json(["The developer profile lists Python and AI agents."])]
        )
        result = CriticAgent("test/model", client).review(
            valid_digest(), valid_profile()
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["unsupported_claims"], [])
        self.assertTrue(
            any("does not contain it" in item for item in result["feedback"])
        )

    def test_paraphrased_claim_is_replaced_by_exact_report_sentence(self):
        client = SequenceClient(
            [
                critic_json(["legacy API removal requires review"]),
                '{"supported": false, "reason": "Review is not in the evidence."}',
            ]
        )
        result = CriticAgent("test/model", client).review(
            valid_digest(), valid_profile()
        )

        self.assertFalse(result["passed"])
        self.assertEqual(
            result["unsupported_claims"],
            ["The legacy API removal requires review."],
        )
        # The focused check saw the exact sentence and its cited evidence.
        check_prompt = client.messages[1][-1]["content"]
        self.assertIn('"The legacy API removal requires review."', check_prompt)
        self.assertIn("EV-0123456789AB", check_prompt)

    def test_claim_verified_as_supported_is_released_and_report_passes(self):
        client = SequenceClient(
            [
                critic_json(["The legacy API removal requires review."]),
                '{"supported": true, "reason": "The evidence says the API is removed."}',
            ]
        )
        result = CriticAgent("test/model", client).review(
            valid_digest(), valid_profile()
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["unsupported_claims"], [])
        self.assertTrue(
            any("Verified as supported" in item for item in result["feedback"])
        )

    def test_low_scores_skip_the_focused_check(self):
        client = SequenceClient(
            [critic_json(["The legacy API removal requires review."], (50, 90, 90))]
        )
        result = CriticAgent("test/model", client).review(
            valid_digest(), valid_profile()
        )

        self.assertFalse(result["passed"])
        self.assertEqual(len(client.messages), 1)

    def test_failed_focused_check_keeps_claim_flagged(self):
        client = SequenceClient(
            [critic_json(["The legacy API removal requires review."]), "no", "no"]
        )
        result = CriticAgent("test/model", client).review(
            valid_digest(), valid_profile()
        )

        self.assertFalse(result["passed"])
        self.assertEqual(len(result["unsupported_claims"]), 1)


class RevisionMustChangeTests(unittest.TestCase):
    def setUp(self):
        self.signal = assessed_signal()
        self.evidence_id = SynthesisAgent.build_evidence_pack(
            [self.signal]
        )[0].evidence_id
        self.flagged = "A relevant compatibility change needs review."
        self.critic = {
            "passed": False,
            "groundedness_score": 60.0,
            "profile_relevance_score": 80.0,
            "completeness_score": 75.0,
            "unsupported_claims": [self.flagged],
            "feedback": ["Be more specific."],
        }

    def _revise(self, responses):
        client = SequenceClient([valid_response(self.evidence_id), *responses])
        agent = SynthesisAgent("test/model", client)
        initial = agent.synthesize([self.signal])
        revised = agent.revise([self.signal], initial, self.critic, revision_number=1)
        return client, revised

    def test_unchanged_revision_is_rejected_then_corrected(self):
        corrected = json.loads(valid_response(self.evidence_id))
        corrected["executive_summary"] = "The legacy API is removed in this release."
        client, revised = self._revise(
            [valid_response(self.evidence_id), json.dumps(corrected)]
        )

        self.assertEqual(revised["executive_summary"], corrected["executive_summary"])
        correction_prompt = client.messages[2][-1]["content"]
        self.assertIn("flagged sentences word for word", correction_prompt)

    def test_flagged_sentence_is_listed_in_revision_prompt(self):
        corrected = json.loads(valid_response(self.evidence_id))
        corrected["executive_summary"] = "The legacy API is removed in this release."
        client, _ = self._revise([json.dumps(corrected)])

        prompt = client.messages[1][-1]["content"]
        self.assertIn("Sentences the critic flagged", prompt)
        self.assertIn(f"- {self.flagged}", prompt)

    def test_revision_that_never_changes_fails_visibly(self):
        same = valid_response(self.evidence_id)
        with self.assertRaises(AnalysisFailedError):
            self._revise([same, same])


class RemoveFlaggedSentenceTests(unittest.TestCase):
    def setUp(self):
        self.signal = assessed_signal()
        self.evidence_id = SynthesisAgent.build_evidence_pack(
            [self.signal]
        )[0].evidence_id
        payload = json.loads(valid_response(self.evidence_id))
        payload["executive_summary"] = (
            "The legacy API is removed. These items offer high actionability."
        )
        client = SequenceClient([json.dumps(payload)])
        self.agent = SynthesisAgent("test/model", client)
        self.digest = self.agent.synthesize([self.signal])

    def test_drop_sentences_keeps_the_rest(self):
        self.assertEqual(
            drop_sentences("A is true. B is overstated! C stays.", ["B is overstated"]),
            "A is true. C stays.",
        )

    def test_flagged_sentence_is_removed_and_nothing_added(self):
        revised = self.agent.remove_flagged_sentences(
            [self.signal], self.digest, ["These items offer high actionability."]
        )
        self.assertEqual(revised["executive_summary"], "The legacy API is removed.")
        self.assertEqual(revised["recommendations"], self.digest["recommendations"])

    def test_removal_that_empties_the_summary_is_refused(self):
        with self.assertRaises(ContractValidationError):
            self.agent.remove_flagged_sentences(
                [self.signal],
                self.digest,
                ["The legacy API is removed.", "These items offer high actionability."],
            )


class WorkflowRemovalFallbackTests(unittest.TestCase):
    def test_failed_rewrite_falls_back_to_removal_then_passes(self):
        signal = assessed_signal()
        evidence_id = SynthesisAgent.build_evidence_pack([signal])[0].evidence_id
        payload = json.loads(valid_response(evidence_id))
        payload["executive_summary"] = (
            "The legacy API is removed. These items offer high actionability."
        )
        flagged = "These items offer high actionability."
        same = json.dumps(payload)
        client = SequenceClient(
            [
                same,  # synthesis
                critic_json([flagged]),  # review 0
                '{"supported": false, "reason": "Not in the evidence."}',
                same,  # revision 1: unchanged
                same,  # revision 1 correction: still unchanged -> fails
                critic_json([]),  # review 1 after code removal
            ]
        )
        workflow = ReviewedSynthesisWorkflow(
            SynthesisAgent("test/model", client),
            CriticAgent("test/model", client),
        )

        result = workflow.run([signal], valid_profile())

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            result["digest"]["executive_summary"], "The legacy API is removed."
        )
        self.assertIsNone(result["revision_failure"])
        self.assertTrue(
            any("removed by code" in item for item in result["critic"]["feedback"])
        )


if __name__ == "__main__":
    unittest.main()
