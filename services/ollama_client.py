"""Ollama Cloud implementation of the provider-neutral LLM interface."""

from __future__ import annotations

from time import perf_counter
from typing import List, Mapping, Optional

from openai import OpenAI

from .llm import LLMRequestError, LLMResult, TokenUsage


OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"


class OllamaRequestError(LLMRequestError):
    """A sanitized Ollama Cloud request failure safe to display in the UI."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ):
        super().__init__(
            message,
            provider="Ollama Cloud",
            status_code=status_code,
            retryable=retryable,
        )


class OllamaClient:
    """Call Ollama Cloud through its OpenAI-compatible chat endpoint."""

    def __init__(self, api_key: str, model: str, sdk_client=None):
        self.model = model
        self._api_key = api_key
        self._client = sdk_client or OpenAI(
            api_key=api_key,
            base_url=OLLAMA_CLOUD_BASE_URL,
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
        response_format,
    ) -> LLMResult:
        selected_model = model or self.model
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
            raise OllamaRequestError(
                self._format_error(status_code, detail),
                status_code=status_code,
                retryable=status_code in {408, 409, 429, 500, 502, 503, 504},
            ) from exc

        latency_ms = round((perf_counter() - started) * 1000)
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise OllamaRequestError(
                "Ollama Cloud returned no assistant message. Try another model."
            ) from exc

        if not isinstance(content, str):
            content = ""
        if not content.strip() and response_format is None:
            raise OllamaRequestError("Ollama Cloud returned an empty response.")

        usage = getattr(response, "usage", None)
        return LLMResult(
            content=content.strip(),
            model=str(getattr(response, "model", selected_model)),
            latency_ms=latency_ms,
            usage=TokenUsage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(
                    getattr(usage, "completion_tokens", 0) or 0
                ),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            ),
        )

    @staticmethod
    def _format_error(status_code: Optional[int], detail: str) -> str:
        if status_code == 401:
            return "Ollama Cloud rejected the API key (401). Check or rotate it."
        if status_code in {402, 403}:
            return (
                "Ollama Cloud rejected the request for account, access, or "
                "credit reasons."
            )
        if status_code == 404:
            return "Ollama Cloud could not find that model (404). Check its ID."
        if status_code == 429:
            return "Ollama Cloud rate limit reached (429). Wait before retrying."
        suffix = f" HTTP {status_code}." if status_code else "."
        return f"Ollama Cloud request failed{suffix} {detail}"

