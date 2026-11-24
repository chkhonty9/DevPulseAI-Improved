from types import SimpleNamespace
import unittest

from services import OpenAIClient


class FakeResponses:
    def __init__(self):
        self.last_request = None

    def create(self, **request):
        self.last_request = request
        return SimpleNamespace(
            output_text='{"status": "ok"}',
            model="openai/test-model",
            usage=SimpleNamespace(
                input_tokens=8,
                output_tokens=3,
                total_tokens=11,
            ),
        )


class OpenAIClientTests(unittest.TestCase):
    def test_json_completion_uses_responses_json_mode(self):
        responses = FakeResponses()
        client = OpenAIClient(
            "sk-test",
            "openai/test-model",
            sdk_client=SimpleNamespace(responses=responses),
        )

        result = client.complete_json(
            [
                {"role": "system", "content": "Return JSON"},
                {"role": "user", "content": "Now"},
            ]
        )

        self.assertEqual(result.content, '{"status": "ok"}')
        self.assertEqual(result.usage.total_tokens, 11)
        self.assertEqual(responses.last_request["model"], "openai/test-model")
        self.assertEqual(
            responses.last_request["text"],
            {"format": {"type": "json_object"}},
        )
        self.assertEqual(responses.last_request["input"][0]["role"], "developer")


if __name__ == "__main__":
    unittest.main()
