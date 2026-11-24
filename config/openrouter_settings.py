"""Resolve OpenRouter settings without exposing API keys.

Local development reads from an environment variable or Streamlit secrets.
The deployed UI can instead provide a user-entered key for the current run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Mapping, Optional


DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class MissingOpenRouterKeyError(ValueError):
    """Raised when the selected key source does not provide a key."""


@dataclass(frozen=True)
class OpenRouterSettings:
    """Validated runtime settings for one OpenRouter client."""

    api_key: str = field(repr=False)
    model: str = DEFAULT_OPENROUTER_MODEL
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    key_source: str = "unknown"
    app_url: str = "http://localhost:8501"
    app_title: str = "DevPulseAI Improved"


def resolve_openrouter_settings(
    *,
    key_mode: str = "local",
    user_api_key: Optional[str] = None,
    model: Optional[str] = None,
    secrets: Optional[Mapping[str, object]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> OpenRouterSettings:
    """Resolve a key for either local configuration or a user session.

    `key_mode="local"` checks `OPENROUTER_API_KEY` and then Streamlit secrets.
    `key_mode="user"` accepts only the key typed by the current user.
    """

    environment = environ if environ is not None else os.environ
    secret_values = secrets or {}
    normalized_mode = key_mode.strip().lower()

    if normalized_mode == "user":
        api_key = (user_api_key or "").strip()
        key_source = "user session"
    elif normalized_mode == "local":
        environment_key = str(environment.get("OPENROUTER_API_KEY", "")).strip()
        secret_key = str(secret_values.get("OPENROUTER_API_KEY", "")).strip()
        api_key = environment_key or secret_key
        key_source = "environment" if environment_key else "Streamlit secrets"
    else:
        raise ValueError("key_mode must be either 'local' or 'user'")

    if not api_key:
        if normalized_mode == "user":
            raise MissingOpenRouterKeyError(
                "Enter an OpenRouter API key for this session."
            )
        raise MissingOpenRouterKeyError(
            "Set OPENROUTER_API_KEY or add it to .streamlit/secrets.toml."
        )

    configured_model = (
        (model or "").strip()
        or str(environment.get("OPENROUTER_MODEL", "")).strip()
        or str(secret_values.get("OPENROUTER_MODEL", "")).strip()
        or DEFAULT_OPENROUTER_MODEL
    )

    return OpenRouterSettings(
        api_key=api_key,
        model=configured_model,
        key_source=key_source,
        app_url=str(environment.get("OPENROUTER_APP_URL", "")).strip()
        or "http://localhost:8501",
    )
