"""Client for the GetQHSE OpenAI-compatible hosted Ollama gateway."""

from __future__ import annotations

from time import perf_counter
from typing import List, Mapping, Optional

from openai import OpenAI

from config import GetQHSESettings
from .llm import LLMRequestError, LLMResult, TokenUsage


class GetQHSERequestError(LLMRequestError):
    """A sanitized GetQHSE gateway failure safe to display in the UI."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ):
        super().__init__(
            message,
            provider="GetQHSE Hosted LLM",
            status_code=status_code,
            retryable=retryable,
        )


class GetQHSEClient:
    """Use the team's hosted models through OpenAI-compatible chat calls."""

    def __init__(self, settings: GetQHSESettings, sdk_client=None):
        self.settings = settings
        self._api_key = settings.api_key
        self._client = sdk_client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=1,
        )

    def list_models(self) -> List[str]:
        """Discover the gateway's current model IDs at runtime."""
        try:
            page = self._client.models.list()
        except Exception as exc:
            self._raise_request_error(exc, action="model discovery")

        model_ids = sorted(
            {
                str(getattr(model, "id", "")).strip()
                for model in getattr(page, "data", [])
                if str(getattr(model, "id", "")).strip()
            }
        )
        if not model_ids:
            raise GetQHSERequestError(
                "GetQHSE model discovery returned no model IDs."
            )
        return model_ids

    def complete_json(
        self,
        messages: List[Mapping[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 400,
    ) -> LLMResult:
        # The team gateway exposes only a subset of the OpenAI API. Prompts
        # request JSON explicitly and Pydantic validates it after the response,
        # so no unsupported response_format option is assumed here.
        return self._complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
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
    ) -> LLMResult:
        selected_model = model or self.settings.model
        request = {
            "model": selected_model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if selected_model == "qwen3.5:2b":
            request["reasoning_effort"] = "none"

        started = perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            self._raise_request_error(exc, action="chat completion")

        latency_ms = round((perf_counter() - started) * 1000)
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise GetQHSERequestError(
                "GetQHSE returned no assistant message."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise GetQHSERequestError("GetQHSE returned an empty response.")

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

    def _raise_request_error(self, exc: Exception, *, action: str) -> None:
        status_code = getattr(exc, "status_code", None)
        detail = str(exc).replace(self._api_key, "[redacted]")
        if status_code == 401:
            message = "GetQHSE rejected the configured API key (401)."
        elif status_code == 400:
            message = "GetQHSE rejected the request or model options (400)."
        elif status_code == 404:
            message = "GetQHSE could not find the selected endpoint or model (404)."
        elif status_code == 429:
            message = "GetQHSE rate limit reached (429). Wait before retrying."
        elif status_code and status_code >= 500:
            message = f"GetQHSE encountered a server error ({status_code})."
        else:
            message = f"GetQHSE {action} failed. {detail}"
        raise GetQHSERequestError(
            message,
            status_code=status_code,
            retryable=status_code in {408, 409, 429, 500, 502, 503, 504},
        ) from exc

