import unittest
from datetime import datetime, timezone

from agents import AnalysisFailedError, RelevanceAgent, RiskAgent
from models import DeveloperProfile
from services import LLMResult, OpenRouterRequestError


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete_json(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return LLMResult(
            content=response,
            model=kwargs.get("model", "test/model"),
            latency_ms=1,
        )


class AgentOpenRouterTests(unittest.TestCase):
    def setUp(self):
        self.signal = {
            "id": "test-signal-1",
            "source": "github",
            "title": "A useful AI library",
            "description": "A developer tool for production AI systems.",
            "url": "https://example.com/signal",
            "metadata": {},
            "collected_at": datetime.now(timezone.utc),
        }
        self.profile = DeveloperProfile(
            technologies=["Python"],
            interests=["AI agents"],
            experience_level="intermediate",
        )

    def test_relevance_agent_uses_injected_llm_client(self):
        client = FakeLLMClient(
            [
                '{"score": 84, "matched_technologies": ["Python"], '
                '"matched_interests": ["AI agents"], '
                '"reasoning": "Directly actionable.", '
                '"actionability": "HIGH"}'
            ]
        )
        agent = RelevanceAgent(model_id="test/model", llm_client=client)

        result = agent.score(self.signal, self.profile)

        self.assertEqual(result["score"], 84)
        self.assertEqual(client.calls[0][1]["model"], "test/model")

    def test_risk_agent_uses_injected_llm_client(self):
        client = FakeLLMClient(
            [
                '{"risk_level": "LOW", "concerns": [], '
                '"evidence_quotes": [], "recommended_actions": [], '
                '"breaking_changes": false}'
            ]
        )
        agent = RiskAgent(model_id="test/model", llm_client=client)

        result = agent.assess(self.signal)

        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(client.calls[0][1]["model"], "test/model")

    def test_malformed_relevance_output_gets_one_correction_attempt(self):
        client = FakeLLMClient(
            [
                "not-json",
                '{"score": 71, "matched_technologies": ["Python"], '
                '"matched_interests": [], '
                '"reasoning": "Corrected output.", '
                '"actionability": "MEDIUM"}',
            ]
        )
        agent = RelevanceAgent(model_id="test/model", llm_client=client)

        result = agent.score(self.signal, self.profile)

        self.assertEqual(result["score"], 71)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("previous response was invalid", client.calls[1][0][-1]["content"])

    def test_relevance_failure_is_visible_after_one_correction(self):
        client = FakeLLMClient(["not-json", '{"score": 999}'])
        agent = RelevanceAgent(model_id="test/model", llm_client=client)

        with self.assertRaises(AnalysisFailedError) as raised:
            agent.score(self.signal, self.profile)

        self.assertEqual(len(client.calls), 2)
        self.assertIn("No heuristic result was substituted", str(raised.exception))

    def test_openrouter_failure_is_not_replaced_with_a_heuristic(self):
        failure = OpenRouterRequestError(
            "OpenRouter rate limit reached (429).",
            status_code=429,
            retryable=True,
        )
        client = FakeLLMClient([failure])
        agent = RiskAgent(model_id="test/model", llm_client=client)

        with self.assertRaises(OpenRouterRequestError):
            agent.assess(self.signal)

        self.assertEqual(len(client.calls), 1)

    def test_invalid_risk_level_is_corrected_once(self):
        client = FakeLLMClient(
            [
                '{"risk_level": "UNKNOWN", "concerns": [], '
                '"evidence_quotes": [], "recommended_actions": [], '
                '"breaking_changes": false}',
                '{"risk_level": "LOW", "concerns": [], '
                '"evidence_quotes": [], "recommended_actions": [], '
                '"breaking_changes": false}',
            ]
        )
        agent = RiskAgent(model_id="test/model", llm_client=client)

        result = agent.assess(self.signal)

        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(len(client.calls), 2)

    def test_unsupported_cve_is_rejected_then_corrected(self):
        client = FakeLLMClient(
            [
                '{"risk_level": "HIGH", '
                '"concerns": ["CVE-2026-99999 affects this library."], '
                '"evidence_quotes": ["developer tool"], '
                '"recommended_actions": ["Patch CVE-2026-99999 immediately."], '
                '"breaking_changes": false}',
                '{"risk_level": "LOW", "concerns": [], '
                '"evidence_quotes": [], "recommended_actions": [], '
                '"breaking_changes": false}',
            ]
        )
        agent = RiskAgent(model_id="test/model", llm_client=client)

        result = agent.assess(self.signal)

        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(len(client.calls), 2)
        self.assertIn(
            "unsupported identifier or version",
            client.calls[1][0][-1]["content"],
        )

    def test_breaking_change_accepts_removes_wording(self):
        signal = {
            **self.signal,
            "title": "Framework removes legacy API",
            "description": "Version 4.0 removes the legacy client API.",
        }
        client = FakeLLMClient(
            [
                '{"risk_level": "MEDIUM", '
                '"concerns": ["The legacy client API is removed."], '
                '"evidence_quotes": ["removes the legacy client API"], '
                '"recommended_actions": ["Review compatibility."], '
                '"breaking_changes": true}'
            ]
        )

        result = RiskAgent("test/model", client).assess(signal)

        self.assertTrue(result["breaking_changes"])
        self.assertEqual(len(client.calls), 1)

    def test_evidence_quote_must_exist_in_signal(self):
        client = FakeLLMClient(
            [
                '{"risk_level": "MEDIUM", '
                '"concerns": ["Potential compatibility problem."], '
                '"evidence_quotes": ["This sentence was invented."], '
                '"recommended_actions": ["Review before adoption."], '
                '"breaking_changes": false}',
                '{"risk_level": "LOW", "concerns": [], '
                '"evidence_quotes": [], "recommended_actions": [], '
                '"breaking_changes": false}',
            ]
        )
        agent = RiskAgent(model_id="test/model", llm_client=client)

        result = agent.assess(self.signal)

        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(len(client.calls), 2)

    def test_unambiguous_object_wrappers_are_normalized(self):
        client = FakeLLMClient(
            [
                '{"risk_level": "MEDIUM", '
                '"concerns": [{"text": "Review production impact."}], '
                '"evidence_quotes": [{"quote": "production AI systems"}], '
                '"recommended_actions": [{"action": "Review before adoption."}], '
                '"breaking_changes": false}'
            ]
        )
        agent = RiskAgent(model_id="test/model", llm_client=client)

        result = agent.assess(self.signal)

        self.assertEqual(result["concerns"], ["Review production impact."])
        self.assertEqual(result["evidence_quotes"], ["production AI systems"])
        self.assertEqual(len(client.calls), 1)

    def test_low_decision_discards_speculative_model_fields(self):
        client = FakeLLMClient(
            [
                '{"risk_level": "LOW", '
                '"concerns": [{"description": "No security policy mentioned."}], '
                '"deprecations": [], "breaking_changes": false}'
            ]
        )
        agent = RiskAgent(model_id="test/model", llm_client=client)

        result = agent.assess(self.signal)

        self.assertEqual(
            result,
            {
                "risk_level": "LOW",
                "concerns": [],
                "evidence_quotes": [],
                "recommended_actions": [],
                "breaking_changes": False,
            },
        )
        self.assertEqual(len(client.calls), 1)

    def test_ambiguous_object_wrapper_is_still_rejected_and_corrected(self):
        client = FakeLLMClient(
            [
                '{"risk_level": "MEDIUM", '
                '"concerns": [{"text": "First", "description": "Second"}], '
                '"evidence_quotes": ["production AI systems"], '
                '"recommended_actions": ["Review before adoption."], '
                '"breaking_changes": false}',
                '{"risk_level": "LOW", "concerns": [], '
                '"evidence_quotes": [], "recommended_actions": [], '
                '"breaking_changes": false}',
            ]
        )
        agent = RiskAgent(model_id="test/model", llm_client=client)

        result = agent.assess(self.signal)

        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(len(client.calls), 2)

    def test_same_signal_uses_profile_specific_context(self):
        client = FakeLLMClient(
            [
                '{"score": 91, "matched_technologies": ["Python"], '
                '"matched_interests": ["AI agents"], '
                '"reasoning": "Matches the Python AI stack.", '
                '"actionability": "HIGH"}',
                '{"score": 34, "matched_technologies": [], '
                '"matched_interests": [], '
                '"reasoning": "Does not match the Rust systems profile.", '
                '"actionability": "LOW"}',
            ]
        )
        agent = RelevanceAgent(model_id="test/model", llm_client=client)
        rust_profile = DeveloperProfile(
            technologies=["Rust"],
            interests=["Distributed systems"],
            experience_level="advanced",
        )

        python_result = agent.score(self.signal, self.profile)
        rust_result = agent.score(self.signal, rust_profile)

        self.assertGreater(python_result["score"], rust_result["score"])
        self.assertIn("Python", client.calls[0][0][-1]["content"])
        self.assertIn("Rust", client.calls[1][0][-1]["content"])


if __name__ == "__main__":
    unittest.main()
