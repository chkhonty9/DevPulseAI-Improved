"""Resolve protected settings for the GetQHSE hosted LLM gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Mapping, Optional


DEFAULT_GETQHSE_BASE_URL = "https://ai-access.getqhse.com/v1"
DEFAULT_GETQHSE_MODEL = "qwen3.5:2b"
DEFAULT_GETQHSE_TIMEOUT_SECONDS = 180.0


class MissingGetQHSEKeyError(ValueError):
    """Raised when protected local configuration has no gateway key."""


@dataclass(frozen=True)
class GetQHSESettings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_GETQHSE_MODEL
    base_url: str = DEFAULT_GETQHSE_BASE_URL
    timeout_seconds: float = DEFAULT_GETQHSE_TIMEOUT_SECONDS
    key_source: str = "unknown"


def resolve_getqhse_settings(
    *,
    model: Optional[str] = None,
    secrets: Optional[Mapping[str, object]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> GetQHSESettings:
    """Resolve GetQHSE settings from backend environment or Streamlit secrets.

    ``GETQHSE_API_KEY`` is preferred. ``OLLAMA_API_KEY`` is supported as an
    alias because that is the variable name used by the team integration guide.
    """

    environment = environ if environ is not None else os.environ
    secret_values = secrets or {}

    environment_key = (
        str(environment.get("GETQHSE_API_KEY", "")).strip()
        or str(environment.get("OLLAMA_API_KEY", "")).strip()
    )
    secret_key = (
        str(secret_values.get("GETQHSE_API_KEY", "")).strip()
        or str(secret_values.get("OLLAMA_API_KEY", "")).strip()
    )
    api_key = environment_key or secret_key
    key_source = "environment" if environment_key else "Streamlit secrets"

    if not api_key:
        raise MissingGetQHSEKeyError(
            "Set GETQHSE_API_KEY (or OLLAMA_API_KEY) in the environment or "
            "add it to .streamlit/secrets.toml."
        )

    configured_model = (
        (model or "").strip()
        or str(environment.get("GETQHSE_MODEL", "")).strip()
        or str(secret_values.get("GETQHSE_MODEL", "")).strip()
        or DEFAULT_GETQHSE_MODEL
    )
    configured_base_url = (
        str(environment.get("GETQHSE_BASE_URL", "")).strip()
        or str(environment.get("AI_BASE_URL", "")).strip()
        or str(secret_values.get("GETQHSE_BASE_URL", "")).strip()
        or str(secret_values.get("AI_BASE_URL", "")).strip()
        or DEFAULT_GETQHSE_BASE_URL
    ).rstrip("/")

    return GetQHSESettings(
        api_key=api_key,
        model=configured_model,
        base_url=configured_base_url,
        key_source=key_source,
    )

