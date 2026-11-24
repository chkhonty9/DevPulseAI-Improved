"""OpenRouter implementation of the provider-neutral LLM interface."""

from __future__ import annotations

from time import perf_counter
from typing import List, Mapping, Optional

from openai import OpenAI

from config import OpenRouterSettings
from .llm import LLMRequestError, LLMResult, TokenUsage


class OpenRouterRequestError(LLMRequestError):
    """A sanitized OpenRouter request failure safe to display in the UI."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ):
        super().__init__(
            message,
            provider="OpenRouter",
            status_code=status_code,
            retryable=retryable,
        )


class OpenRouterClient:
    """Call OpenRouter through its OpenAI-compatible chat API."""

    def __init__(self, settings: OpenRouterSettings, sdk_client=None):
        self.settings = settings
        self._api_key = settings.api_key
        self._client = sdk_client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=60.0,
            max_retries=1,
            default_headers={
                "HTTP-Referer": settings.app_url,
                "X-Title": settings.app_title,
            },
        )

    def complete_json(
        self,
        messages: List[Mapping[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 400,
    ) -> LLMResult:
        return self._complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    def complete_text(
        self,
        messages: List[Mapping[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 400,
    ) -> LLMResult:
        return self._complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
        )

    def test_connection(self) -> LLMResult:
        """Make a minimal explicit request for the UI connection test."""
        return self.complete_text(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0,
            # Some free routed models spend part of this budget on reasoning
            # before emitting visible text. Eight tokens can therefore return
            # an otherwise successful response with no displayable content.
            max_tokens=128,
        )

    def _complete(
        self,
        messages: List[Mapping[str, str]],
        *,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        response_format,
    ) -> LLMResult:
        selected_model = model or self.settings.model
        request = {
            "model": selected_model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            request["response_format"] = response_format

        started = perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            detail = str(exc).replace(self._api_key, "[redacted]")
            retryable = status_code in {
                408,
                409,
                429,
                500,
                502,
                503,
                504,
                524,
                529,
            }
            raise OpenRouterRequestError(
                self._format_request_error(status_code, detail),
                status_code=status_code,
                retryable=retryable,
            ) from exc

        latency_ms = round((perf_counter() - started) * 1000)
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise OpenRouterRequestError(
                "OpenRouter returned no assistant message. Try another model."
            ) from exc

        if not isinstance(content, str):
            content = ""
        if not content.strip() and response_format is None:
            raise OpenRouterRequestError("OpenRouter returned an empty response.")

        usage = getattr(response, "usage", None)
        token_usage = TokenUsage(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )

        return LLMResult(
            content=content.strip(),
            model=str(getattr(response, "model", selected_model)),
            latency_ms=latency_ms,
            usage=token_usage,
        )

    @staticmethod
    def _format_request_error(status_code: Optional[int], detail: str) -> str:
        if status_code == 401:
            return "OpenRouter rejected the API key (401). Check or rotate the key."
        if status_code == 402:
            return (
                "OpenRouter rejected the request for credit or key-limit reasons "
                "(402). Check the key's limits and account balance."
            )
        if status_code == 429:
            return (
                "OpenRouter rate limit reached (429). Free models have strict "
                "limits; wait for the limit to reset before trying again."
            )
        if status_code in {503, 529}:
            return (
                "The selected OpenRouter model is temporarily unavailable "
                f"({status_code}). Try again later or choose another model."
            )

        suffix = f" HTTP {status_code}." if status_code else "."
        return f"OpenRouter request failed{suffix} {detail}"
