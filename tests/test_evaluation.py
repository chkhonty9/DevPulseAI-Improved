import json
from pathlib import Path
import re
import unittest

from evaluations import EvaluationRunner, load_cases
from services import LLMResult, TokenUsage


CASES_PATH = Path(__file__).resolve().parents[1] / "evaluations" / "cases.json"


class EvaluationFakeClient:
    def complete_json(self, messages, **kwargs):
        system = messages[0]["content"]
        user = messages[-1]["content"]

        if "score technical signals" in system:
            payload = {
                "score": 90.0,
                "matched_technologies": [],
                "matched_interests": [],
                "reasoning": "The signal directly affects the supplied stack.",
                "actionability": "HIGH",
            }
        elif "Assess only explicit technical risk facts" in system:
            if "CVE-2026-12345" in user:
                payload = {
                    "risk_level": "CRITICAL",
                    "concerns": ["A critical authentication bypass is reported."],
                    "evidence_quotes": ["critical authentication bypass"],
                    "recommended_actions": ["Review the advisory immediately."],
                    "breaking_changes": False,
                }
            else:
                payload = {
                    "risk_level": "MEDIUM",
                    "concerns": ["The legacy client API is removed."],
                    "evidence_quotes": ["removes the legacy client API"],
                    "recommended_actions": ["Review migration before upgrading."],
                    "breaking_changes": True,
                }
        elif "synthesize developer intelligence" in system:
            evidence_ids = list(
                dict.fromkeys(re.findall(r"EV-[A-F0-9]{12}", user))
            )
            payload = {
                "executive_summary": "The cited API removal requires review.",
                "summary_evidence_ids": [evidence_ids[0]],
                "priority_evidence_ids": [evidence_ids[0]],
                "recommendations": [
                    {
                        "text": "Review migration before upgrading.",
                        "evidence_ids": [evidence_ids[0]],
                    }
                ],
            }
        elif "independent evidence and relevance critic" in system:
            payload = {
                "groundedness_score": 95.0,
                "profile_relevance_score": 90.0,
                "completeness_score": 90.0,
                "unsupported_claims": [],
                "feedback": ["The report is supported by its evidence."],
            }
        else:
            raise AssertionError(f"Unexpected evaluation prompt: {system}")

        return LLMResult(
            content=json.dumps(payload),
            model=kwargs.get("model") or "test/model",
            latency_ms=10,
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


class EvaluationTests(unittest.TestCase):
    def test_manual_dataset_has_18_unique_valid_cases(self):
        cases = load_cases(CASES_PATH)

        self.assertEqual(len(cases), 18)
        self.assertEqual(len({case.id for case in cases}), 18)

    def test_runner_calculates_reproducible_metrics(self):
        cases = load_cases(CASES_PATH)[:2]
        runner = EvaluationRunner(
            EvaluationFakeClient(),
            "test/model",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
        )

        report = runner.run(cases)
        metrics = report["metrics"]

        self.assertEqual(metrics["relevance_agreement_percent"], 100.0)
        self.assertEqual(metrics["risk_agreement_percent"], 100.0)
        self.assertEqual(metrics["invalid_output_percent"], 0.0)
        self.assertEqual(metrics["groundedness_percent"], 100.0)
        self.assertEqual(metrics["model_call_count"], 6)
        self.assertEqual(metrics["prompt_tokens"], 60)
        self.assertEqual(metrics["completion_tokens"], 30)
        self.assertEqual(metrics["estimated_cost_usd"], 0.00012)
        self.assertEqual(report["pipeline_status"], "completed")


if __name__ == "__main__":
    unittest.main()
