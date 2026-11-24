from types import SimpleNamespace
import unittest

from services import OllamaClient, OllamaRequestError


class FakeCompletions:
    def __init__(self):
        self.last_request = None

    def create(self, **request):
        self.last_request = request
        return SimpleNamespace(
            model="deepseek-v4.1-flash",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"status": "ok"}')
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=3,
                total_tokens=13,
            ),
        )


class StatusFailure(RuntimeError):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class FailingCompletions:
    def create(self, **request):
        raise StatusFailure(401, "rejected ollama-secret")


class OllamaClientTests(unittest.TestCase):
    def test_json_completion_uses_ollama_model_and_json_mode(self):
        completions = FakeCompletions()
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        client = OllamaClient(
            "ollama-test-key",
            "deepseek-v4.1-flash",
            sdk_client=sdk_client,
        )

        result = client.complete_json(
            [{"role": "user", "content": "Return JSON"}]
        )

        self.assertEqual(result.content, '{"status": "ok"}')
        self.assertEqual(result.usage.total_tokens, 13)
        self.assertEqual(
            completions.last_request["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            completions.last_request["model"], "deepseek-v4.1-flash"
        )

    def test_error_is_actionable_and_redacts_key(self):
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FailingCompletions())
        )
        client = OllamaClient(
            "ollama-secret",
            "deepseek-v4.1-flash",
            sdk_client=sdk_client,
        )

        with self.assertRaises(OllamaRequestError) as raised:
            client.complete_text([{"role": "user", "content": "Hello"}])

        self.assertEqual(raised.exception.status_code, 401)
        self.assertNotIn("ollama-secret", str(raised.exception))
        self.assertIn("rejected", str(raised.exception).lower())


if __name__ == "__main__":
    unittest.main()
