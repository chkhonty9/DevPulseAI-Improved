import unittest

from config import MissingOpenRouterKeyError, resolve_openrouter_settings


class OpenRouterSettingsTests(unittest.TestCase):
    def test_local_environment_key_has_precedence(self):
        settings = resolve_openrouter_settings(
            key_mode="local",
            secrets={"OPENROUTER_API_KEY": "secret-key"},
            environ={
                "OPENROUTER_API_KEY": "environment-key",
                "OPENROUTER_MODEL": "provider/environment-model",
            },
        )

        self.assertEqual(settings.api_key, "environment-key")
        self.assertEqual(settings.key_source, "environment")
        self.assertEqual(settings.model, "provider/environment-model")

    def test_user_mode_does_not_fall_back_to_local_key(self):
        with self.assertRaises(MissingOpenRouterKeyError):
            resolve_openrouter_settings(
                key_mode="user",
                user_api_key="",
                secrets={"OPENROUTER_API_KEY": "local-secret"},
                environ={"OPENROUTER_API_KEY": "environment-secret"},
            )

    def test_user_key_is_not_in_settings_representation(self):
        settings = resolve_openrouter_settings(
            key_mode="user",
            user_api_key="session-secret-key",
            environ={},
        )

        self.assertNotIn("session-secret-key", repr(settings))


if __name__ == "__main__":
    unittest.main()
