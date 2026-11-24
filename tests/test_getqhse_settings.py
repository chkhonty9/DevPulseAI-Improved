import unittest

from config import (
    DEFAULT_GETQHSE_BASE_URL,
    MissingGetQHSEKeyError,
    resolve_getqhse_settings,
)


class GetQHSESettingsTests(unittest.TestCase):
    def test_dedicated_environment_key_has_precedence(self):
        settings = resolve_getqhse_settings(
            secrets={"GETQHSE_API_KEY": "secret-key"},
            environ={
                "GETQHSE_API_KEY": "environment-key",
                "GETQHSE_MODEL": "qwen3:8b",
            },
        )

        self.assertEqual(settings.api_key, "environment-key")
        self.assertEqual(settings.key_source, "environment")
        self.assertEqual(settings.model, "qwen3:8b")
        self.assertEqual(settings.base_url, DEFAULT_GETQHSE_BASE_URL)
        self.assertEqual(settings.timeout_seconds, 180.0)

    def test_team_guide_variable_names_are_supported(self):
        settings = resolve_getqhse_settings(
            secrets={},
            environ={
                "OLLAMA_API_KEY": "team-guide-key",
                "AI_BASE_URL": "https://example.test/v1/",
            },
        )

        self.assertEqual(settings.api_key, "team-guide-key")
        self.assertEqual(settings.base_url, "https://example.test/v1")

    def test_missing_key_is_visible(self):
        with self.assertRaises(MissingGetQHSEKeyError):
            resolve_getqhse_settings(secrets={}, environ={})

    def test_key_is_not_in_settings_representation(self):
        settings = resolve_getqhse_settings(
            secrets={"GETQHSE_API_KEY": "protected-team-key"},
            environ={},
        )

        self.assertNotIn("protected-team-key", repr(settings))


if __name__ == "__main__":
    unittest.main()
