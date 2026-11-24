"""Provider-neutral language-model interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Protocol


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    latency_ms: int
    usage: TokenUsage = field(default_factory=TokenUsage)


class LLMRequestError(RuntimeError):
    """A sanitized provider request error safe to show in the UI."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class LLMClient(Protocol):
    """The small interface required by DevPulseAI reasoning agents."""

    def complete_json(
        self,
        messages: List[Mapping[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 400,
    ) -> LLMResult:
        """Return a non-streaming response whose content is JSON text."""

    def complete_text(
        self,
        messages: List[Mapping[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 400,
    ) -> LLMResult:
        """Return a non-streaming text response."""

    def test_connection(self) -> LLMResult:
        """Make a minimal provider request for an explicit UI test."""
