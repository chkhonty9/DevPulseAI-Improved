import unittest

from services import (
    LLMRequestError,
    LLMResult,
    ObservabilityLLMClient,
    TokenUsage,
)


class SuccessfulClient:
    def complete_json(self, messages, **kwargs):
        return LLMResult(
            content="{}",
            model=kwargs.get("model") or "test/model",
            latency_ms=25,
            usage=TokenUsage(
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
            ),
        )


class FailingClient:
    def complete_json(self, messages, **kwargs):
        raise LLMRequestError(
            "Provider unavailable.",
            provider="test",
            retryable=True,
        )


class ObservabilityTests(unittest.TestCase):
    def test_records_stage_usage_and_latency_without_content(self):
        client = ObservabilityLLMClient(SuccessfulClient())
        client.complete_json(
            [
                {
                    "role": "system",
                    "content": "You score technical signals for developers.",
                },
                {"role": "user", "content": "secret prompt value"},
            ],
            model="test/model",
        )

        snapshot = client.snapshot()

        self.assertEqual(snapshot["call_count"], 1)
        self.assertEqual(snapshot["total_tokens"], 15)
        self.assertEqual(snapshot["by_stage"][0]["stage"], "relevance")
        serialized = str(snapshot)
        self.assertNotIn("secret prompt value", serialized)
        self.assertNotIn("content", snapshot["calls"][0])

    def test_failed_call_is_recorded_and_reraised(self):
        client = ObservabilityLLMClient(FailingClient())

        with self.assertRaises(LLMRequestError):
            client.complete_json(
                [
                    {
                        "role": "system",
                        "content": "Assess only explicit technical risk facts.",
                    }
                ],
                model="test/model",
            )

        snapshot = client.snapshot()
        self.assertEqual(snapshot["failed_call_count"], 1)
        self.assertEqual(snapshot["by_stage"][0]["stage"], "risk")
        self.assertEqual(
            snapshot["calls"][0]["error_type"],
            "LLMRequestError",
        )


if __name__ == "__main__":
    unittest.main()
