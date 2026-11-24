import unittest

from services import (
    GeminiClient,
    OllamaClient,
    OpenAIClient,
    OpenRouterClient,
    ProviderConfigurationError,
    create_provider_client,
    validate_provider_key,
)


class ProviderTests(unittest.TestCase):
    def test_gemini_key_is_not_sent_to_openrouter(self):
        with self.assertRaisesRegex(ProviderConfigurationError, "Google Gemini"):
            validate_provider_key("OpenRouter", "AIza-example")

    def test_openrouter_key_is_not_sent_to_gemini(self):
        with self.assertRaisesRegex(ProviderConfigurationError, "OpenRouter"):
            validate_provider_key("Google Gemini", "sk-or-v1-example")

    def test_factory_builds_the_selected_provider(self):
        clients = (
            create_provider_client(
                "OpenRouter", api_key="sk-or-v1-example", model="openrouter/free"
            ),
            create_provider_client(
                "OpenAI", api_key="sk-example", model="gpt-6-luna"
            ),
            create_provider_client(
                "Google Gemini", api_key="AIza-example", model="gemini-3.8-flash"
            ),
            create_provider_client(
                "Ollama Cloud",
                api_key="ollama-example-key",
                model="deepseek-v4.1-flash",
            ),
        )

        self.assertIsInstance(clients[0], OpenRouterClient)
        self.assertIsInstance(clients[1], OpenAIClient)
        self.assertIsInstance(clients[2], GeminiClient)
        self.assertIsInstance(clients[3], OllamaClient)

    def test_openai_key_is_not_sent_to_ollama(self):
        with self.assertRaisesRegex(ProviderConfigurationError, "OpenAI"):
            validate_provider_key("Ollama Cloud", "sk-example")


if __name__ == "__main__":
    unittest.main()
