"""External service integrations used by DevPulseAI."""

from .llm import LLMClient, LLMRequestError, LLMResult, TokenUsage
from .observability import LLMCallObservation, ObservabilityLLMClient
from .getqhse_client import GetQHSEClient, GetQHSERequestError
from .gemini_client import GeminiClient, GeminiRequestError
from .openai_client import OpenAIClient, OpenAIRequestError
from .ollama_client import (
    OLLAMA_CLOUD_BASE_URL,
    OllamaClient,
    OllamaRequestError,
)
from .openrouter_client import OpenRouterClient, OpenRouterRequestError
from .providers import (
    PROVIDERS,
    ProviderConfigurationError,
    ProviderDefinition,
    create_provider_client,
    validate_provider_key,
)

__all__ = [
    "LLMClient",
    "LLMRequestError",
    "LLMResult",
    "TokenUsage",
    "LLMCallObservation",
    "ObservabilityLLMClient",
    "GetQHSEClient",
    "GetQHSERequestError",
    "OpenRouterClient",
    "OpenRouterRequestError",
    "OpenAIClient",
    "OpenAIRequestError",
    "OllamaClient",
    "OllamaRequestError",
    "OLLAMA_CLOUD_BASE_URL",
    "GeminiClient",
    "GeminiRequestError",
    "PROVIDERS",
    "ProviderDefinition",
    "ProviderConfigurationError",
    "create_provider_client",
    "validate_provider_key",
]
