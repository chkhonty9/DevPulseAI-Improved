"""Secret-safe runtime observability for provider-neutral LLM calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, List, Mapping, Optional

from .llm import LLMClient, LLMResult


@dataclass(frozen=True)
class LLMCallObservation:
    stage: str
    operation: str
    model: str
    provider_latency_ms: int
    wall_latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    failed: bool
    error_type: Optional[str] = None


class ObservabilityLLMClient:
    """Record call metadata without retaining prompts, responses, or keys."""

    def __init__(self, delegate: LLMClient):
        self.delegate = delegate
        self.observations: List[LLMCallObservation] = []

    def complete_json(self, messages, **kwargs) -> LLMResult:
        return self._observe(
            "complete_json",
            self._classify_stage(messages),
            self.delegate.complete_json,
            messages,
            requested_model=kwargs.get("model"),
            **kwargs,
        )

    def complete_text(self, messages, **kwargs) -> LLMResult:
        return self._observe(
            "complete_text",
            self._classify_stage(messages),
            self.delegate.complete_text,
            messages,
            requested_model=kwargs.get("model"),
            **kwargs,
        )

    def test_connection(self) -> LLMResult:
        return self._observe(
            "test_connection",
            "connection",
            self.delegate.test_connection,
            requested_model=None,
        )

    def _observe(
        self,
        operation_name: str,
        stage: str,
        operation: Callable[..., LLMResult],
        *args,
        requested_model: Optional[str],
        **kwargs,
    ) -> LLMResult:
        started = perf_counter()
        try:
            result = operation(*args, **kwargs)
        except Exception as exc:
            self.observations.append(
                LLMCallObservation(
                    stage=stage,
                    operation=operation_name,
                    model=requested_model or "unknown",
                    provider_latency_ms=0,
                    wall_latency_ms=round((perf_counter() - started) * 1000),
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    failed=True,
                    error_type=type(exc).__name__,
                )
            )
            raise

        self.observations.append(
            LLMCallObservation(
                stage=stage,
                operation=operation_name,
                model=result.model,
                provider_latency_ms=result.latency_ms,
                wall_latency_ms=round((perf_counter() - started) * 1000),
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                failed=False,
            )
        )
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return aggregate and per-stage metadata safe for UI display."""
        stage_totals: dict[str, dict[str, Any]] = {}
        for observation in self.observations:
            row = stage_totals.setdefault(
                observation.stage,
                {
                    "stage": observation.stage,
                    "calls": 0,
                    "failed_calls": 0,
                    "provider_latency_ms": 0,
                    "wall_latency_ms": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            row["calls"] += 1
            row["failed_calls"] += int(observation.failed)
            row["provider_latency_ms"] += observation.provider_latency_ms
            row["wall_latency_ms"] += observation.wall_latency_ms
            row["prompt_tokens"] += observation.prompt_tokens
            row["completion_tokens"] += observation.completion_tokens
            row["total_tokens"] += observation.total_tokens

        return {
            "call_count": len(self.observations),
            "failed_call_count": sum(item.failed for item in self.observations),
            "provider_latency_ms": sum(
                item.provider_latency_ms for item in self.observations
            ),
            "wall_latency_ms": sum(
                item.wall_latency_ms for item in self.observations
            ),
            "prompt_tokens": sum(
                item.prompt_tokens for item in self.observations
            ),
            "completion_tokens": sum(
                item.completion_tokens for item in self.observations
            ),
            "total_tokens": sum(item.total_tokens for item in self.observations),
            "by_stage": list(stage_totals.values()),
            "calls": [asdict(item) for item in self.observations],
        }

    @staticmethod
    def _classify_stage(messages: List[Mapping[str, str]]) -> str:
        system_text = " ".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        ).casefold()
        if "score technical signals" in system_text:
            return "relevance"
        if "explicit technical risk facts" in system_text:
            return "risk"
        if "synthesize developer intelligence" in system_text:
            return "synthesis"
        if "independent evidence and relevance critic" in system_text:
            return "critic"
        if "revise developer intelligence" in system_text:
            return "revision"
        return "other"
