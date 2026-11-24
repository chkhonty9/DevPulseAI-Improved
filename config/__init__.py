"""Application configuration helpers."""

from .getqhse_settings import (
    DEFAULT_GETQHSE_BASE_URL,
    DEFAULT_GETQHSE_MODEL,
    DEFAULT_GETQHSE_TIMEOUT_SECONDS,
    GetQHSESettings,
    MissingGetQHSEKeyError,
    resolve_getqhse_settings,
)

from .openrouter_settings import (
    DEFAULT_OPENROUTER_MODEL,
    MissingOpenRouterKeyError,
    OpenRouterSettings,
    resolve_openrouter_settings,
)

__all__ = [
    "DEFAULT_GETQHSE_BASE_URL",
    "DEFAULT_GETQHSE_MODEL",
    "DEFAULT_GETQHSE_TIMEOUT_SECONDS",
    "GetQHSESettings",
    "MissingGetQHSEKeyError",
    "resolve_getqhse_settings",
    "DEFAULT_OPENROUTER_MODEL",
    "MissingOpenRouterKeyError",
    "OpenRouterSettings",
    "resolve_openrouter_settings",
]
