"""Direct OpenAI implementation of the provider-neutral LLM interface."""

from __future__ import annotations

from time import perf_counter
from typing import List, Mapping, Optional

from openai import OpenAI

from .llm import LLMRequestError, LLMResult, TokenUsage


class OpenAIRequestError(LLMRequestError):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ):
        super().__init__(
            message,
            provider="OpenAI",
            status_code=status_code,
            retryable=retryable,
        )


class OpenAIClient:
    """Call OpenAI directly through the Responses API."""

    def __init__(self, api_key: str, model: str, sdk_client=None):
        self.model = model
        self._api_key = api_key
        self._client = sdk_client or OpenAI(
            api_key=api_key,
            timeout=60.0,
            max_retries=1,
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
            max_tokens=max_tokens,
            json_mode=True,
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
            max_tokens=max_tokens,
            json_mode=False,
        )

    def test_connection(self) -> LLMResult:
        return self.complete_text(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=128,
        )

    def _complete(
        self,
        messages: List[Mapping[str, str]],
        *,
        model: Optional[str],
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult:
        selected_model = model or self.model
        input_messages = [
            {
                "role": "developer" if message.get("role") == "system" else message.get("role", "user"),
                "content": message.get("content", ""),
            }
            for message in messages
        ]
        request = {
            "model": selected_model,
            "input": input_messages,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            request["text"] = {"format": {"type": "json_object"}}

        started = perf_counter()
        try:
            response = self._client.responses.create(**request)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            detail = str(exc).replace(self._api_key, "[redacted]")
            raise OpenAIRequestError(
                self._format_error(status_code, detail),
                status_code=status_code,
                retryable=status_code in {408, 409, 429, 500, 502, 503, 504},
            ) from exc

        latency_ms = round((perf_counter() - started) * 1000)
        content = str(getattr(response, "output_text", "") or "").strip()
        if not content and not json_mode:
            raise OpenAIRequestError("OpenAI returned an empty response.")

        usage = getattr(response, "usage", None)
        return LLMResult(
            content=content,
            model=str(getattr(response, "model", selected_model)),
            latency_ms=latency_ms,
            usage=TokenUsage(
                prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            ),
        )

    @staticmethod
    def _format_error(status_code: Optional[int], detail: str) -> str:
        if status_code == 401:
            return "OpenAI rejected the API key (401). Check or rotate the key."
        if status_code == 429:
            return "OpenAI rate limit reached (429). Wait before trying again."
        if status_code in {402, 403}:
            return "OpenAI rejected the request for account or billing reasons."
        return f"OpenAI request failed. {detail}"
