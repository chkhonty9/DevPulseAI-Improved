from types import SimpleNamespace
import unittest

from config import OpenRouterSettings
from services import OpenRouterClient, OpenRouterRequestError


class FakeCompletions:
    def __init__(self):
        self.last_request = None

    def create(self, **request):
        self.last_request = request
        return SimpleNamespace(
            model="provider/test-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"status": "ok"}')
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=4,
                total_tokens=15,
            ),
        )


class FailingCompletions:
    def create(self, **request):
        raise RuntimeError("request rejected for test-key")


class StatusFailure(RuntimeError):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class StatusFailingCompletions:
    def __init__(self, status_code):
        self.status_code = status_code

    def create(self, **request):
        raise StatusFailure(self.status_code, "provider details")


class OpenRouterClientTests(unittest.TestCase):
    def test_json_completion_uses_openrouter_model_and_json_mode(self):
        completions = FakeCompletions()
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        settings = OpenRouterSettings(
            api_key="test-key",
            model="provider/test-model",
        )
        client = OpenRouterClient(settings, sdk_client=sdk_client)

        result = client.complete_json(
            [{"role": "user", "content": "Return JSON"}]
        )

        self.assertEqual(result.content, '{"status": "ok"}')
        self.assertEqual(result.usage.total_tokens, 15)
        self.assertEqual(
            completions.last_request["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            completions.last_request["model"], "provider/test-model"
        )

    def test_request_error_redacts_api_key(self):
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FailingCompletions())
        )
        settings = OpenRouterSettings(
            api_key="test-key",
            model="provider/test-model",
        )
        client = OpenRouterClient(settings, sdk_client=sdk_client)

        with self.assertRaises(OpenRouterRequestError) as raised:
            client.complete_text([{"role": "user", "content": "Hello"}])

        self.assertNotIn("test-key", str(raised.exception))
        self.assertIn("[redacted]", str(raised.exception))

    def test_rate_limit_error_is_actionable_and_marked_retryable(self):
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=StatusFailingCompletions(429)
            )
        )
        settings = OpenRouterSettings(
            api_key="test-key",
            model="provider/test-model",
        )
        client = OpenRouterClient(settings, sdk_client=sdk_client)

        with self.assertRaises(OpenRouterRequestError) as raised:
            client.complete_text([{"role": "user", "content": "Hello"}])

        self.assertEqual(raised.exception.status_code, 429)
        self.assertTrue(raised.exception.retryable)
        self.assertIn("rate limit", str(raised.exception).lower())


if __name__ == "__main__":
    unittest.main()
