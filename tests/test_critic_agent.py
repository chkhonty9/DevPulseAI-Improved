import json
from datetime import datetime, timezone
import unittest

from agents import AnalysisFailedError, CriticAgent
from models import DeveloperProfile
from services import LLMResult


EVIDENCE_ID = "EV-0123456789AB"


def valid_digest():
    signal = {
        "id": "signal-1",
        "source": "github",
        "title": "Framework removes its legacy API",
        "description": "The legacy API is removed.",
        "url": "https://example.com/change",
        "metadata": {},
        "collected_at": datetime.now(timezone.utc),
        "relevance": {
            "score": 88.0,
            "matched_technologies": ["Python"],
            "matched_interests": ["AI agents"],
            "reasoning": "Relevant to the developer profile.",
            "actionability": "HIGH",
        },
        "risk": {
            "risk_level": "MEDIUM",
            "concerns": ["The legacy API is removed."],
            "evidence_quotes": ["legacy API is removed"],
            "recommended_actions": ["Review compatibility."],
            "breaking_changes": True,
        },
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_signals": 1,
        "executive_summary": "The legacy API removal requires review.",
        "summary_evidence_ids": [EVIDENCE_ID],
        "priority_signals": [dict(signal)],
        "signals_by_source": {"github": [signal]},
        "recommendations": [
            {
                "text": "Review compatibility before upgrading.",
                "evidence_ids": [EVIDENCE_ID],
            }
        ],
        "evidence_pack": [
            {
                "evidence_id": EVIDENCE_ID,
                "signal_id": "signal-1",
                "source": "github",
                "title": "Framework removes its legacy API",
                "claim": (
                    "Framework removes its legacy API: "
                    "The legacy API is removed."
                ),
                "source_url": "https://example.com/change",
                "relevance_score": 88.0,
                "risk_level": "MEDIUM",
            }
        ],
    }


def valid_profile():
    return DeveloperProfile(
        technologies=["Python"],
        interests=["AI agents"],
        experience_level="intermediate",
    )


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


def critic_response(**overrides):
    payload = {
        "groundedness_score": 92.0,
        "profile_relevance_score": 86.0,
        "completeness_score": 84.0,
        "unsupported_claims": [],
        "feedback": ["Keep citations next to each action."],
    }
    payload.update(overrides)
    return json.dumps(payload)


class CriticAgentTests(unittest.TestCase):
    def test_valid_digest_is_reviewed_by_ai_and_passes(self):
        client = SequenceClient([critic_response()])
        result = CriticAgent("test/model", client).review(
            valid_digest(),
            valid_profile(),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(client.calls, 1)

    def test_unknown_evidence_id_fails_before_ai_call(self):
        digest = valid_digest()
        digest["recommendations"][0]["evidence_ids"] = ["EV-FFFFFFFFFFFF"]
        client = SequenceClient([])

        result = CriticAgent("test/model", client).review(
            digest,
            valid_profile(),
        )

        self.assertFalse(result["passed"])
        self.assertEqual(client.calls, 0)
        self.assertIn("Unknown evidence IDs", result["unsupported_claims"][0])

    def test_unknown_priority_signal_fails_before_ai_call(self):
        digest = valid_digest()
        digest["priority_signals"][0]["id"] = "unknown-signal"
        client = SequenceClient([])

        result = CriticAgent("test/model", client).review(
            digest,
            valid_profile(),
        )

        self.assertFalse(result["passed"])
        self.assertEqual(client.calls, 0)
        self.assertTrue(
            any(
                "Unknown priority signal IDs" in claim
                for claim in result["unsupported_claims"]
            )
        )

    def test_low_score_or_unsupported_claim_needs_review(self):
        client = SequenceClient(
            [
                critic_response(
                    groundedness_score=65.0,
                    unsupported_claims=["An unsupported factual claim."],
                )
            ]
        )

        result = CriticAgent("test/model", client).review(
            valid_digest(),
            valid_profile(),
        )

        self.assertFalse(result["passed"])

    def test_invalid_critic_output_gets_one_correction(self):
        client = SequenceClient(
            [
                '{"groundedness_score": 200}',
                critic_response(),
            ]
        )

        result = CriticAgent("test/model", client).review(
            valid_digest(),
            valid_profile(),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(client.calls, 2)

    def test_invalid_critic_output_twice_stops_visibly(self):
        client = SequenceClient(
            ['{"groundedness_score": 200}', '{"groundedness_score": 200}']
        )

        with self.assertRaises(AnalysisFailedError):
            CriticAgent("test/model", client).review(
                valid_digest(),
                valid_profile(),
            )

        self.assertEqual(client.calls, 2)


if __name__ == "__main__":
    unittest.main()
