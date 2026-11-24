from types import SimpleNamespace
import unittest

from config import GetQHSESettings
from services import GetQHSEClient, GetQHSERequestError


class FakeCompletions:
    def __init__(self):
        self.last_request = None

    def create(self, **request):
        self.last_request = request
        return SimpleNamespace(
            model="qwen3.5:2b",
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"status":"ok"}'))
            ],
            usage=SimpleNamespace(
                prompt_tokens=9,
                completion_tokens=3,
                total_tokens=12,
            ),
        )


class FakeModels:
    def list(self):
        return SimpleNamespace(
            data=[
                SimpleNamespace(id="qwen3:8b"),
                SimpleNamespace(id="qwen3.5:2b"),
            ]
        )


class StatusFailure(RuntimeError):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class FailingCompletions:
    def create(self, **request):
        raise StatusFailure(401, "rejected protected-team-key")


class GetQHSEClientTests(unittest.TestCase):
    def make_settings(self):
        return GetQHSESettings(
            api_key="protected-team-key",
            model="qwen3.5:2b",
        )

    def test_chat_uses_verified_deployment_options(self):
        completions = FakeCompletions()
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
            models=FakeModels(),
        )
        client = GetQHSEClient(self.make_settings(), sdk_client=sdk_client)

        result = client.complete_json(
            [{"role": "user", "content": "Return JSON"}]
        )

        self.assertEqual(result.content, '{"status":"ok"}')
        self.assertEqual(result.usage.total_tokens, 12)
        self.assertEqual(completions.last_request["model"], "qwen3.5:2b")
        self.assertEqual(completions.last_request["reasoning_effort"], "none")
        self.assertFalse(completions.last_request["stream"])
        self.assertNotIn("response_format", completions.last_request)

    def test_models_are_discovered_and_sorted(self):
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions()),
            models=FakeModels(),
        )
        client = GetQHSEClient(self.make_settings(), sdk_client=sdk_client)

        self.assertEqual(client.list_models(), ["qwen3.5:2b", "qwen3:8b"])

    def test_gateway_error_redacts_key(self):
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FailingCompletions()),
            models=FakeModels(),
        )
        client = GetQHSEClient(self.make_settings(), sdk_client=sdk_client)

        with self.assertRaises(GetQHSERequestError) as raised:
            client.complete_text([{"role": "user", "content": "Hello"}])

        self.assertEqual(raised.exception.status_code, 401)
        self.assertNotIn("protected-team-key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
