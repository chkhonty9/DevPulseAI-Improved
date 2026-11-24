from copy import deepcopy
from datetime import datetime, timezone
import json
import unittest

from pydantic import ValidationError

from agents import AnalysisFailedError, SynthesisAgent
from services import LLMResult


def assessed_signal():
    return {
        "id": "signal-1",
        "source": "github",
        "title": "Framework removes its legacy API",
        "description": "The legacy API is removed and projects must review compatibility.",
        "url": "https://example.com/framework-change",
        "metadata": {},
        "collected_at": datetime.now(timezone.utc),
        "relevance": {
            "score": 88.0,
            "matched_technologies": ["Python"],
            "matched_interests": ["AI agents"],
            "reasoning": "The change affects the developer's stack.",
            "actionability": "HIGH",
        },
        "risk": {
            "risk_level": "MEDIUM",
            "concerns": ["The legacy API is removed."],
            "evidence_quotes": ["legacy API is removed"],
            "recommended_actions": ["Review compatibility before upgrading."],
            "breaking_changes": True,
        },
    }


class SequenceClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete_json(self, messages, **kwargs):
        self.calls += 1
        return LLMResult(
            content=next(self.responses),
            model=kwargs.get("model") or "test/model",
            latency_ms=1,
        )


def valid_response(evidence_id):
    return json.dumps(
        {
            "executive_summary": "A relevant compatibility change needs review.",
            "summary_evidence_ids": [evidence_id],
            "priority_evidence_ids": [evidence_id],
            "recommendations": [
                {
                    "text": "Review compatibility before adopting the change.",
                    "evidence_ids": [evidence_id],
                }
            ],
        }
    )


class RecordingClient(SequenceClient):
    def __init__(self, responses):
        super().__init__(responses)
        self.messages = []

    def complete_json(self, messages, **kwargs):
        self.messages.append(messages)
        return super().complete_json(messages, **kwargs)


class ProfileAndPriorityTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "technologies": ["Python"],
            "interests": ["AI agents"],
            "experience_level": "intermediate",
        }

    def _signal(self, signal_id, score, risk_level="LOW"):
        signal = deepcopy(assessed_signal())
        signal["id"] = signal_id
        signal["title"] = f"Signal {signal_id}"
        signal["url"] = f"https://example.com/{signal_id}"
        signal["relevance"]["score"] = float(score)
        if risk_level == "LOW":
            signal["risk"] = {
                "risk_level": "LOW",
                "concerns": [],
                "evidence_quotes": [],
                "recommended_actions": [],
                "breaking_changes": False,
            }
        else:
            signal["risk"]["risk_level"] = risk_level
        return signal

    def test_profile_and_matches_reach_the_synthesis_prompt(self):
        signal = self._signal("a", 88)
        evidence_id = SynthesisAgent.build_evidence_pack([signal])[0].evidence_id
        client = RecordingClient([valid_response(evidence_id)])

        SynthesisAgent("test/model", client).synthesize([signal], self.profile)

        prompt = client.messages[0][-1]["content"]
        self.assertIn("Developer profile", prompt)
        self.assertIn("- Technologies: Python", prompt)
        self.assertIn("- Interests: AI agents", prompt)
        self.assertIn('"matched_interests": [\n      "AI agents"', prompt)
        self.assertIn("Focus rules", prompt)

    def test_priority_is_ranked_by_scores_not_model_choice(self):
        signals = [
            self._signal("low", 40),
            self._signal("top", 90),
            self._signal("risky", 70, "HIGH"),  # 70 x 1.5 = 105
            self._signal("mid", 75),
        ]
        pack = SynthesisAgent.build_evidence_pack(signals)
        low_id = next(item.evidence_id for item in pack if item.signal_id == "low")
        # The model names only the least relevant item as its priority.
        client = RecordingClient([valid_response(low_id)])

        digest = SynthesisAgent("test/model", client).synthesize(
            signals, self.profile
        )

        self.assertEqual(
            [signal["id"] for signal in digest["priority_signals"]],
            ["risky", "top", "mid", "low"],
        )

    def test_priority_is_capped_at_five(self):
        signals = [self._signal(f"s{i}", 50 + i) for i in range(7)]
        pack = SynthesisAgent.build_evidence_pack(signals)
        client = RecordingClient([valid_response(pack[0].evidence_id)])

        digest = SynthesisAgent("test/model", client).synthesize(signals)

        self.assertEqual(len(digest["priority_signals"]), 5)
        self.assertEqual(digest["priority_signals"][0]["id"], "s6")


class SynthesisAgentTests(unittest.TestCase):
    def setUp(self):
        self.signal = assessed_signal()
        self.evidence_id = SynthesisAgent.build_evidence_pack(
            [self.signal]
        )[0].evidence_id

    def test_synthesis_calls_ai_and_preserves_valid_citations(self):
        client = SequenceClient([valid_response(self.evidence_id)])
        digest = SynthesisAgent(
            model_id="test/model",
            llm_client=client,
        ).synthesize([self.signal])

        self.assertEqual(client.calls, 1)
        self.assertEqual(digest["summary_evidence_ids"], [self.evidence_id])
        self.assertEqual(
            digest["recommendations"][0]["evidence_ids"],
            [self.evidence_id],
        )
        self.assertEqual(
            digest["evidence_pack"][0]["evidence_id"],
            self.evidence_id,
        )

    def test_unknown_evidence_id_gets_one_correction_attempt(self):
        client = SequenceClient(
            [
                valid_response("EV-FFFFFFFFFFFF"),
                valid_response(self.evidence_id),
            ]
        )
        digest = SynthesisAgent(
            model_id="test/model",
            llm_client=client,
        ).synthesize([self.signal])

        self.assertEqual(client.calls, 2)
        self.assertEqual(digest["summary_evidence_ids"], [self.evidence_id])

    def test_recommendation_without_citation_is_rejected(self):
        invalid = json.dumps(
            {
                "executive_summary": "A compatibility change needs review.",
                "summary_evidence_ids": [self.evidence_id],
                "priority_evidence_ids": [self.evidence_id],
                "recommendations": [
                    {"text": "Review compatibility.", "evidence_ids": []}
                ],
            }
        )
        client = SequenceClient([invalid, invalid])

        with self.assertRaises(AnalysisFailedError):
            SynthesisAgent(
                model_id="test/model",
                llm_client=client,
            ).synthesize([self.signal])

        self.assertEqual(client.calls, 2)

    def test_evidence_id_is_stable_and_item_is_frozen(self):
        first = SynthesisAgent.build_evidence_pack([self.signal])[0]
        second = SynthesisAgent.build_evidence_pack([self.signal])[0]

        self.assertEqual(first.evidence_id, second.evidence_id)
        with self.assertRaises(ValidationError):
            first.claim = "Changed claim"

    def test_revision_uses_critic_feedback_and_same_evidence(self):
        revised_payload = json.loads(valid_response(self.evidence_id))
        revised_payload["executive_summary"] = (
            "The cited API removal requires a compatibility review."
        )
        client = SequenceClient(
            [
                valid_response(self.evidence_id),
                json.dumps(revised_payload),
            ]
        )
        agent = SynthesisAgent(model_id="test/model", llm_client=client)
        initial = agent.synthesize([self.signal])
        critic = {
            "passed": False,
            "groundedness_score": 60.0,
            "profile_relevance_score": 80.0,
            "completeness_score": 75.0,
            "unsupported_claims": ["The first summary was too broad."],
            "feedback": ["Limit the summary to the cited API removal."],
        }

        revised = agent.revise(
            [self.signal],
            initial,
            critic,
            revision_number=1,
        )

        self.assertEqual(client.calls, 2)
        self.assertEqual(
            revised["executive_summary"],
            revised_payload["executive_summary"],
        )
        self.assertEqual(
            revised["evidence_pack"],
            initial["evidence_pack"],
        )

    def test_partial_revision_preserves_validated_recommendations(self):
        partial_revision = json.dumps(
            {
                "executive_summary": "A narrower cited compatibility summary.",
                "summary_evidence_ids": [self.evidence_id],
            }
        )
        client = SequenceClient(
            [valid_response(self.evidence_id), partial_revision]
        )
        agent = SynthesisAgent(model_id="test/model", llm_client=client)
        initial = agent.synthesize([self.signal])
        critic = {
            "passed": False,
            "groundedness_score": 60.0,
            "profile_relevance_score": 80.0,
            "completeness_score": 75.0,
            "unsupported_claims": ["The first summary was too broad."],
            "feedback": ["Narrow the summary."],
        }

        revised = agent.revise(
            [self.signal],
            initial,
            critic,
            revision_number=1,
        )

        self.assertEqual(
            revised["recommendations"],
            initial["recommendations"],
        )
        self.assertEqual(
            revised["executive_summary"],
            "A narrower cited compatibility summary.",
        )


if __name__ == "__main__":
    unittest.main()
