"""Provider definitions, key hints, and client construction."""

from __future__ import annotations

from dataclasses import dataclass

from config import OpenRouterSettings
from .gemini_client import GeminiClient
from .llm import LLMClient
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient
from .openrouter_client import OpenRouterClient


@dataclass(frozen=True)
class ProviderDefinition:
    name: str
    key_label: str
    key_placeholder: str
    default_model: str
    model_help: str


PROVIDERS = {
    "OpenRouter": ProviderDefinition(
        name="OpenRouter",
        key_label="OpenRouter API key",
        key_placeholder="sk-or-v1-…",
        default_model="openrouter/free",
        model_help="Use an OpenRouter provider/model slug.",
    ),
    "OpenAI": ProviderDefinition(
        name="OpenAI",
        key_label="OpenAI API key",
        key_placeholder="sk-…",
        default_model="gpt-6-luna",
        model_help="Use an OpenAI API model ID.",
    ),
    "Google Gemini": ProviderDefinition(
        name="Google Gemini",
        key_label="Gemini API key",
        key_placeholder="AIza…",
        default_model="gemini-3.8-flash",
        model_help="Use a Gemini API model ID from Google AI Studio.",
    ),
    "Ollama Cloud": ProviderDefinition(
        name="Ollama Cloud",
        key_label="Ollama Cloud API key",
        key_placeholder="Paste an Ollama API key…",
        default_model="deepseek-v4.1-flash",
        model_help="Use a cloud model ID listed on ollama.com.",
    ),
}


class ProviderConfigurationError(ValueError):
    """The selected provider and supplied key are clearly incompatible."""


def validate_provider_key(provider: str, api_key: str) -> str:
    key = (api_key or "").strip()
    if not key:
        raise ProviderConfigurationError(
            f"Enter a {PROVIDERS[provider].key_label} for this session."
        )

    if provider == "OpenRouter":
        if key.startswith("AIza"):
            raise ProviderConfigurationError(
                "This looks like a Gemini key. Select Google Gemini as the provider."
            )
        if key.startswith("sk-") and not key.startswith("sk-or-"):
            raise ProviderConfigurationError(
                "This looks like an OpenAI key. Select OpenAI as the provider."
            )
    elif provider == "OpenAI":
        if key.startswith("AIza"):
            raise ProviderConfigurationError(
                "This looks like a Gemini key. Select Google Gemini as the provider."
            )
        if key.startswith("sk-or-"):
            raise ProviderConfigurationError(
                "This looks like an OpenRouter key. Select OpenRouter as the provider."
            )
    elif provider == "Google Gemini" and key.startswith("sk-"):
        suggested = "OpenRouter" if key.startswith("sk-or-") else "OpenAI"
        raise ProviderConfigurationError(
            f"This looks like a {suggested} key. Select {suggested} as the provider."
        )
    elif provider == "Ollama Cloud":
        if key.startswith("AIza"):
            raise ProviderConfigurationError(
                "This looks like a Gemini key. Select Google Gemini as the provider."
            )
        if key.startswith("sk-"):
            suggested = "OpenRouter" if key.startswith("sk-or-") else "OpenAI"
            raise ProviderConfigurationError(
                f"This looks like a {suggested} key. Select {suggested} as the provider."
            )
    return key


def create_provider_client(
    provider: str,
    *,
    api_key: str,
    model: str,
) -> LLMClient:
    if provider not in PROVIDERS:
        raise ProviderConfigurationError(f"Unsupported AI provider: {provider}")

    key = validate_provider_key(provider, api_key)
    selected_model = (model or "").strip() or PROVIDERS[provider].default_model

    if provider == "OpenRouter":
        return OpenRouterClient(
            OpenRouterSettings(
                api_key=key,
                model=selected_model,
                key_source="user session",
            )
        )
    if provider == "OpenAI":
        return OpenAIClient(key, selected_model)
    if provider == "Google Gemini":
        return GeminiClient(key, selected_model)
    return OllamaClient(key, selected_model)
