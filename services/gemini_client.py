"""Direct Google Gemini implementation of the LLM interface."""

from __future__ import annotations

from time import perf_counter
from typing import List, Mapping, Optional

from google import genai
from google.genai import types

from .llm import LLMRequestError, LLMResult, TokenUsage


class GeminiRequestError(LLMRequestError):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ):
        super().__init__(
            message,
            provider="Google Gemini",
            status_code=status_code,
            retryable=retryable,
        )


class GeminiClient:
    """Call Google Gemini directly with the official Google Gen AI SDK."""

    def __init__(self, api_key: str, model: str, sdk_client=None):
        self.model = model
        self._api_key = api_key
        self._client = sdk_client or genai.Client(api_key=api_key)

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
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
        )

    def test_connection(self) -> LLMResult:
        return self.complete_text(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0,
            max_tokens=128,
        )

    def _complete(
        self,
        messages: List[Mapping[str, str]],
        *,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult:
        selected_model = model or self.model
        system_instruction = "\n\n".join(
            message.get("content", "")
            for message in messages
            if message.get("role") == "system"
        )
        contents = "\n\n".join(
            f"{message.get('role', 'user').upper()}: {message.get('content', '')}"
            for message in messages
            if message.get("role") != "system"
        )
        config = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else "text/plain",
        )

        started = perf_counter()
        try:
            response = self._client.models.generate_content(
                model=selected_model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            status_code = getattr(exc, "code", None) or getattr(
                exc, "status_code", None
            )
            detail = str(exc).replace(self._api_key, "[redacted]")
            raise GeminiRequestError(
                self._format_error(status_code, detail),
                status_code=status_code,
                retryable=status_code in {408, 429, 500, 502, 503, 504},
            ) from exc

        latency_ms = round((perf_counter() - started) * 1000)
        content = str(getattr(response, "text", "") or "").strip()
        if not content and not json_mode:
            raise GeminiRequestError("Google Gemini returned an empty response.")

        usage = getattr(response, "usage_metadata", None)
        return LLMResult(
            content=content,
            model=str(getattr(response, "model_version", selected_model)),
            latency_ms=latency_ms,
            usage=TokenUsage(
                prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                completion_tokens=int(
                    getattr(usage, "candidates_token_count", 0) or 0
                ),
                total_tokens=int(getattr(usage, "total_token_count", 0) or 0),
            ),
        )

    @staticmethod
    def _format_error(status_code: Optional[int], detail: str) -> str:
        if status_code in {400, 401, 403}:
            return (
                "Google Gemini rejected the API key or project access. "
                "Check the key in Google AI Studio."
            )
        if status_code == 429:
            return "Google Gemini rate limit reached (429). Wait before trying again."
        return f"Google Gemini request failed. {detail}"
