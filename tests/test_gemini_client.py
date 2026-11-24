from types import SimpleNamespace
import unittest

from services import GeminiClient


class FakeModels:
    def __init__(self):
        self.last_request = None

    def generate_content(self, **request):
        self.last_request = request
        return SimpleNamespace(
            text='{"status": "ok"}',
            model_version="gemini-test",
            usage_metadata=SimpleNamespace(
                prompt_token_count=7,
                candidates_token_count=2,
                total_token_count=9,
            ),
        )


class GeminiClientTests(unittest.TestCase):
    def test_json_completion_requests_json_mime_type(self):
        models = FakeModels()
        client = GeminiClient(
            "AIza-test",
            "gemini-test",
            sdk_client=SimpleNamespace(models=models),
        )

        result = client.complete_json(
            [
                {"role": "system", "content": "Return JSON"},
                {"role": "user", "content": "Now"},
            ]
        )

        self.assertEqual(result.content, '{"status": "ok"}')
        self.assertEqual(result.usage.total_tokens, 9)
        self.assertEqual(models.last_request["model"], "gemini-test")
        self.assertEqual(
            models.last_request["config"].response_mime_type,
            "application/json",
        )
        self.assertEqual(
            models.last_request["config"].system_instruction,
            "Return JSON",
        )


if __name__ == "__main__":
    unittest.main()
