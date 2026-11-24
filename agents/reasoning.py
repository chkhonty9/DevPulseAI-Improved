"""Shared controls for honest, bounded AI reasoning."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

from services import LLMClient


class AgentOutputError(ValueError):
    """The model returned content that does not satisfy the agent contract."""


class AnalysisFailedError(RuntimeError):
    """An analysis could not produce valid AI output after one correction."""


def parse_json_object(content: str) -> Dict[str, Any]:
    """Parse a JSON object, accepting an optional Markdown code fence."""
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if not text:
        raise AgentOutputError("the model returned an empty response")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentOutputError("the response was not valid JSON") from exc

    if not isinstance(payload, dict):
        raise AgentOutputError("the JSON response must be an object")
    return payload


def complete_validated_json(
    *,
    llm_client: LLMClient,
    messages: List[Mapping[str, str]],
    validator: Callable[[Dict[str, Any]], Dict[str, Any]],
    task_name: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Request and validate JSON, allowing one correction attempt only."""
    attempt_messages = list(messages)
    last_error: Optional[AgentOutputError] = None

    for attempt in range(2):
        response = llm_client.complete_json(
            attempt_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            return validator(parse_json_object(response.content))
        except AgentOutputError as exc:
            last_error = exc
            if attempt == 1:
                break

            attempt_messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": response.content or "",
                },
                {
                    "role": "user",
                    "content": (
                        "Your previous response was invalid because "
                        f"{exc}. Return one corrected JSON object only."
                    ),
                },
            ]

    raise AnalysisFailedError(
        f"{task_name} failed after one correction attempt: {last_error}. "
        "No heuristic result was substituted."
    )
