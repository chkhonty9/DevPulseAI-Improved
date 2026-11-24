from copy import deepcopy
import unittest

from agents import AnalysisFailedError
from tests.test_critic_agent import valid_digest, valid_profile
from workflows import ReviewedSynthesisWorkflow


def critic_result(passed):
    return {
        "passed": passed,
        "groundedness_score": 90.0 if passed else 60.0,
        "profile_relevance_score": 90.0 if passed else 65.0,
        "completeness_score": 90.0 if passed else 68.0,
        "unsupported_claims": [] if passed else ["Unsupported claim."],
        "feedback": ["Remove or support the claim."],
    }


class FakeSynthesisAgent:
    def __init__(self):
        self.synthesize_calls = 0
        self.revise_calls = 0

    def synthesize(self, signals):
        self.synthesize_calls += 1
        return valid_digest()

    def revise(
        self,
        signals,
        previous_digest,
        critic,
        *,
        revision_number,
    ):
        self.revise_calls += 1
        revised = deepcopy(previous_digest)
        revised["executive_summary"] = f"Revised summary {revision_number}."
        return revised


class SequenceCriticAgent:
    def __init__(self, verdicts):
        self.verdicts = iter(verdicts)
        self.calls = 0

    def review(self, digest, profile):
        self.calls += 1
        return critic_result(next(self.verdicts))


class FailingRevisionSynthesisAgent(FakeSynthesisAgent):
    def revise(
        self,
        signals,
        previous_digest,
        critic,
        *,
        revision_number,
    ):
        self.revise_calls += 1
        raise AnalysisFailedError("The model omitted required revision fields.")


def assessed_signals():
    return list(valid_digest()["signals_by_source"]["github"])


class ReviewedSynthesisWorkflowTests(unittest.TestCase):
    def test_first_passing_review_needs_no_revision(self):
        synthesis = FakeSynthesisAgent()
        critic = SequenceCriticAgent([True])
        workflow = ReviewedSynthesisWorkflow(synthesis, critic)

        result = workflow.run(assessed_signals(), valid_profile())

        self.assertEqual(result["status"], "completed")
        self.assertEqual(synthesis.synthesize_calls, 1)
        self.assertEqual(synthesis.revise_calls, 0)
        self.assertEqual(len(result["revision_history"]), 1)

    def test_report_can_pass_after_two_revisions(self):
        synthesis = FakeSynthesisAgent()
        critic = SequenceCriticAgent([False, False, True])
        workflow = ReviewedSynthesisWorkflow(synthesis, critic)

        result = workflow.run(assessed_signals(), valid_profile())

        self.assertEqual(result["status"], "completed")
        self.assertEqual(synthesis.revise_calls, 2)
        self.assertEqual(critic.calls, 3)
        self.assertEqual(
            [item["revision_number"] for item in result["revision_history"]],
            [0, 1, 2],
        )

    def test_failing_report_stops_after_exactly_two_revisions(self):
        synthesis = FakeSynthesisAgent()
        critic = SequenceCriticAgent([False, False, False])
        workflow = ReviewedSynthesisWorkflow(synthesis, critic)

        result = workflow.run(assessed_signals(), valid_profile())

        self.assertEqual(result["status"], "needs_review")
        self.assertFalse(result["critic"]["passed"])
        self.assertEqual(synthesis.revise_calls, 2)
        self.assertEqual(critic.calls, 3)
        self.assertEqual(len(result["revision_history"]), 3)
        self.assertEqual(result["revision_attempts"], 2)

    def test_invalid_revision_returns_needs_review_instead_of_crashing(self):
        synthesis = FailingRevisionSynthesisAgent()
        critic = SequenceCriticAgent([False])
        workflow = ReviewedSynthesisWorkflow(synthesis, critic)

        result = workflow.run(assessed_signals(), valid_profile())

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["revision_attempts"], 1)
        self.assertIsNotNone(result["revision_failure"])
        self.assertEqual(synthesis.revise_calls, 1)
        self.assertEqual(critic.calls, 1)

    def test_existing_digest_is_not_synthesized_twice(self):
        synthesis = FakeSynthesisAgent()
        critic = SequenceCriticAgent([True])
        workflow = ReviewedSynthesisWorkflow(synthesis, critic)

        workflow.run(
            assessed_signals(),
            valid_profile(),
            initial_digest=valid_digest(),
        )

        self.assertEqual(synthesis.synthesize_calls, 0)


if __name__ == "__main__":
    unittest.main()
