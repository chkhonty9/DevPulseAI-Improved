from datetime import datetime, timezone
import unittest

from pydantic import ValidationError

from agents import AnalysisFailedError, RelevanceAgent, SignalCollector
from models import (
    CitedRecommendation,
    CriticResult,
    DeveloperProfile,
    EvidenceItem,
    IntelligenceDigest,
    PipelineResult,
    RelevanceResult,
    RiskResult,
    Signal,
    ContractValidationError,
)
from services import LLMResult


def valid_signal_payload():
    return {
        "id": "signal-1",
        "source": "github",
        "title": "Validated signal",
        "description": "A complete signal payload.",
        "url": "https://example.com/signal-1",
        "metadata": {},
        "collected_at": datetime.now(timezone.utc),
    }


class SequenceClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete_json(self, messages, **kwargs):
        self.calls += 1
        return LLMResult(
            content=next(self.responses),
            model="test/model",
            latency_ms=1,
        )


class ContractTests(unittest.TestCase):
    def test_all_planned_contracts_can_validate(self):
        profile = DeveloperProfile(
            technologies=["Python"],
            interests=["AI agents"],
            experience_level="advanced",
        )
        relevance = RelevanceResult(
            score=82.0,
            matched_technologies=["Python"],
            matched_interests=["AI agents"],
            reasoning="Actionable.",
            actionability="HIGH",
        )
        risk = RiskResult(
            risk_level="LOW",
            concerns=[],
            evidence_quotes=[],
            recommended_actions=[],
            breaking_changes=False,
        )
        signal = Signal(
            **valid_signal_payload(),
            relevance=relevance,
            risk=risk,
        )
        evidence = EvidenceItem(
            evidence_id="EV-0123456789AB",
            signal_id=signal.id,
            source=signal.source,
            title=signal.title,
            claim="The source describes a validated signal.",
            source_url=signal.url,
            relevance_score=relevance.score,
            risk_level=risk.risk_level,
        )
        digest = IntelligenceDigest(
            generated_at=datetime.now(timezone.utc),
            total_signals=1,
            executive_summary="One signal analyzed.",
            summary_evidence_ids=[evidence.evidence_id],
            priority_signals=[signal],
            signals_by_source={"github": [signal]},
            recommendations=[
                CitedRecommendation(
                    text="Review the signal.",
                    evidence_ids=[evidence.evidence_id],
                )
            ],
            evidence_pack=[evidence],
        )
        critic = CriticResult(
            passed=True,
            groundedness_score=100.0,
            profile_relevance_score=90.0,
            completeness_score=95.0,
            unsupported_claims=[],
            feedback=[],
        )
        result = PipelineResult(
            status="completed",
            signals=[signal],
            digest=digest,
            critic=critic,
            evidence=[evidence],
        )

        self.assertEqual(profile.experience_level, "advanced")
        self.assertEqual(result.digest.total_signals, 1)

    def test_unknown_agent_output_field_is_rejected_after_one_correction(self):
        client = SequenceClient(
            [
                '{"score": 80, "reasoning": "Relevant.", "unknown": true}',
                '{"score": 80, "reasoning": "Still invalid.", "unknown": true}',
            ]
        )
        agent = RelevanceAgent(model_id="test/model", llm_client=client)

        with self.assertRaises(AnalysisFailedError) as raised:
            agent.score(
                valid_signal_payload(),
                DeveloperProfile(
                    technologies=["Python"],
                    interests=["AI agents"],
                    experience_level="advanced",
                ),
            )

        self.assertEqual(client.calls, 2)
        self.assertIn("Extra inputs are not permitted", str(raised.exception))

    def test_missing_signal_id_fails_visibly(self):
        raw_signal = valid_signal_payload()
        raw_signal.pop("id")

        with self.assertRaises(ContractValidationError) as raised:
            SignalCollector().collect([raw_signal])

        self.assertIn("id", str(raised.exception))

    def test_invalid_score_type_is_not_coerced(self):
        with self.assertRaises(ValidationError):
            RelevanceResult(
                score="82",
                matched_technologies=[],
                matched_interests=[],
                reasoning="Wrong type",
                actionability="LOW",
            )

    def test_breaking_change_cannot_be_low_risk(self):
        with self.assertRaises(ValidationError):
            RiskResult(
                risk_level="LOW",
                concerns=["The legacy API was removed."],
                evidence_quotes=["removed the legacy API"],
                recommended_actions=["Review the migration guide."],
                breaking_changes=True,
            )

    def test_low_risk_cannot_include_speculative_concerns(self):
        with self.assertRaises(ValidationError):
            RiskResult(
                risk_level="LOW",
                concerns=["No security documentation was mentioned."],
                evidence_quotes=["A community-driven project."],
                recommended_actions=["Review the source code."],
                breaking_changes=False,
            )

    def test_digest_rejects_unknown_evidence_citation(self):
        relevance = RelevanceResult(
            score=82.0,
            matched_technologies=["Python"],
            matched_interests=[],
            reasoning="Relevant.",
            actionability="HIGH",
        )
        risk = RiskResult(
            risk_level="LOW",
            concerns=[],
            evidence_quotes=[],
            recommended_actions=[],
            breaking_changes=False,
        )
        signal = Signal(**valid_signal_payload(), relevance=relevance, risk=risk)
        evidence = EvidenceItem(
            evidence_id="EV-0123456789AB",
            signal_id=signal.id,
            source=signal.source,
            title=signal.title,
            claim="Validated source claim.",
            source_url=signal.url,
            relevance_score=relevance.score,
            risk_level=risk.risk_level,
        )

        with self.assertRaises(ValidationError):
            IntelligenceDigest(
                generated_at=datetime.now(timezone.utc),
                total_signals=1,
                executive_summary="One signal analyzed.",
                summary_evidence_ids=["EV-FFFFFFFFFFFF"],
                priority_signals=[signal],
                signals_by_source={"github": [signal]},
                recommendations=[
                    CitedRecommendation(
                        text="Review the signal.",
                        evidence_ids=[evidence.evidence_id],
                    )
                ],
                evidence_pack=[evidence],
            )


if __name__ == "__main__":
    unittest.main()
