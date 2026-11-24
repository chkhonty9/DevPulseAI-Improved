"""Metrics-driven evaluation runner for the DevPulseAI reasoning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, TypeVar

from pydantic import Field, model_validator

from agents import (
    AnalysisFailedError,
    CriticAgent,
    RelevanceAgent,
    RiskAgent,
    SynthesisAgent,
)
from models import ContractModel, DeveloperProfile, Signal
from services import LLMClient, LLMRequestError, LLMResult
from workflows import ReviewedSynthesisWorkflow


class ExpectedOutcome(ContractModel):
    relevance_min: float = Field(ge=0, le=100)
    relevance_max: float = Field(ge=0, le=100)
    risk_level: str
    breaking_changes: bool

    @model_validator(mode="after")
    def require_ordered_range(self) -> "ExpectedOutcome":
        if self.relevance_min > self.relevance_max:
            raise ValueError("relevance_min must not exceed relevance_max")
        if self.risk_level not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            raise ValueError("invalid expected risk level")
        return self


class EvaluationCase(ContractModel):
    id: str = Field(min_length=1)
    signal: Signal
    profile: DeveloperProfile
    expected: ExpectedOutcome


@dataclass
class RecordedCall:
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    failed_request: bool


class RecordingLLMClient:
    """Provider-neutral proxy that records timing, usage, and request errors."""

    def __init__(self, delegate: LLMClient):
        self.delegate = delegate
        self.calls: List[RecordedCall] = []

    def complete_json(self, messages, **kwargs):
        return self._record(self.delegate.complete_json, messages, **kwargs)

    def complete_text(self, messages, **kwargs):
        return self._record(self.delegate.complete_text, messages, **kwargs)

    def test_connection(self):
        return self._record(self.delegate.test_connection)

    def _record(self, operation: Callable, *args, **kwargs):
        started = perf_counter()
        try:
            result = operation(*args, **kwargs)
        except Exception:
            self.calls.append(
                RecordedCall(
                    latency_ms=round((perf_counter() - started) * 1000),
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    failed_request=True,
                )
            )
            raise

        self.calls.append(
            RecordedCall(
                latency_ms=result.latency_ms,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                failed_request=False,
            )
        )
        return result


ResultType = TypeVar("ResultType")


class EvaluationRunner:
    """Run reviewed cases and return reproducible aggregate metrics."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_id: str,
        *,
        input_cost_per_million: Optional[float] = None,
        output_cost_per_million: Optional[float] = None,
    ):
        self.client = RecordingLLMClient(llm_client)
        self.model_id = model_id
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.invalid_responses = 0
        self.request_errors = 0

    def run(self, cases: List[EvaluationCase]) -> Dict[str, Any]:
        relevance = RelevanceAgent(self.model_id, self.client)
        risk = RiskAgent(self.model_id, self.client)
        case_results = []
        assessed_cases = []
        grounding_checks = 0
        grounded_checks = 0

        for case in cases:
            signal = case.signal.model_dump(mode="python", exclude_none=True)
            relevance_result, relevance_error = self._run_operation(
                lambda: relevance.score(signal, case.profile)
            )
            risk_result, risk_error = self._run_operation(
                lambda: risk.assess(signal)
            )

            relevance_agrees = None
            if relevance_result is not None:
                score = relevance_result["score"]
                relevance_agrees = (
                    case.expected.relevance_min
                    <= score
                    <= case.expected.relevance_max
                )

            risk_agrees = None
            if risk_result is not None:
                risk_agrees = (
                    risk_result["risk_level"] == case.expected.risk_level
                    and risk_result["breaking_changes"]
                    == case.expected.breaking_changes
                )
                source_text = self._normalize_text(
                    f"{signal['title']} {signal['description']}"
                )
                for quote in risk_result["evidence_quotes"]:
                    grounding_checks += 1
                    if self._normalize_text(quote) in source_text:
                        grounded_checks += 1

            if relevance_result is not None and risk_result is not None:
                assessed_cases.append(
                    (
                        {
                            **signal,
                            "relevance": relevance_result,
                            "risk": risk_result,
                        },
                        case.profile,
                    )
                )

            case_results.append(
                {
                    "case_id": case.id,
                    "relevance_score": relevance_result.get("score")
                    if relevance_result
                    else None,
                    "expected_relevance_range": [
                        case.expected.relevance_min,
                        case.expected.relevance_max,
                    ],
                    "relevance_agrees": relevance_agrees,
                    "risk_level": risk_result.get("risk_level")
                    if risk_result
                    else None,
                    "expected_risk_level": case.expected.risk_level,
                    "risk_agrees": risk_agrees,
                    "errors": [
                        error
                        for error in (relevance_error, risk_error)
                        if error is not None
                    ],
                }
            )

        pipeline_result = None
        pipeline_error = None
        if assessed_cases:
            signal, profile = assessed_cases[0]
            synthesis = SynthesisAgent(self.model_id, self.client)
            critic = CriticAgent(self.model_id, self.client)
            workflow = ReviewedSynthesisWorkflow(synthesis, critic)
            pipeline_result, pipeline_error = self._run_operation(
                lambda: workflow.run([signal], profile),
                count_invalid_retries=False,
            )

        if pipeline_result is not None:
            evidence_ids = {
                item["evidence_id"] for item in pipeline_result["evidence"]
            }
            cited_ids = set(
                pipeline_result["digest"]["summary_evidence_ids"]
            )
            for recommendation in pipeline_result["digest"]["recommendations"]:
                cited_ids.update(recommendation["evidence_ids"])
            for evidence_id in cited_ids:
                grounding_checks += 1
                if evidence_id in evidence_ids:
                    grounded_checks += 1

        valid_relevance = [
            item for item in case_results if item["relevance_agrees"] is not None
        ]
        valid_risk = [
            item for item in case_results if item["risk_agrees"] is not None
        ]
        successful_responses = sum(
            1 for call in self.client.calls if not call.failed_request
        )
        total_responses = successful_responses
        prompt_tokens = sum(call.prompt_tokens for call in self.client.calls)
        completion_tokens = sum(
            call.completion_tokens for call in self.client.calls
        )
        estimated_cost = self._estimate_cost(prompt_tokens, completion_tokens)

        revision_attempts = (
            pipeline_result["revision_attempts"] if pipeline_result else 0
        )
        revision_success_rate = None
        if revision_attempts:
            revision_success_rate = (
                100.0 if pipeline_result["status"] == "completed" else 0.0
            )

        return {
            "model": self.model_id,
            "case_count": len(cases),
            "case_results": case_results,
            "metrics": {
                "relevance_agreement_percent": self._percentage(
                    sum(item["relevance_agrees"] for item in valid_relevance),
                    len(valid_relevance),
                ),
                "risk_agreement_percent": self._percentage(
                    sum(item["risk_agrees"] for item in valid_risk),
                    len(valid_risk),
                ),
                "invalid_output_percent": self._percentage(
                    self.invalid_responses,
                    total_responses,
                ),
                "request_error_count": self.request_errors,
                "groundedness_percent": self._percentage(
                    grounded_checks,
                    grounding_checks,
                    empty_value=100.0,
                ),
                "model_call_count": len(self.client.calls),
                "latency_ms_total": sum(
                    call.latency_ms for call in self.client.calls
                ),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": sum(
                    call.total_tokens for call in self.client.calls
                ),
                "estimated_cost_usd": estimated_cost,
                "revision_attempts": revision_attempts,
                "revision_success_percent": revision_success_rate,
            },
            "pipeline_status": pipeline_result["status"]
            if pipeline_result
            else "not_run",
            "pipeline_error": pipeline_error,
        }

    def _run_operation(
        self,
        operation: Callable[[], ResultType],
        *,
        count_invalid_retries: bool = True,
    ) -> tuple[Optional[ResultType], Optional[str]]:
        calls_before = len(self.client.calls)
        try:
            result = operation()
        except (AnalysisFailedError, LLMRequestError, ValueError) as exc:
            calls_used = len(self.client.calls) - calls_before
            if isinstance(exc, AnalysisFailedError):
                self.invalid_responses += (
                    max(1, calls_used) if count_invalid_retries else 1
                )
            if isinstance(exc, LLMRequestError):
                self.request_errors += 1
            return None, str(exc)

        calls_used = len(self.client.calls) - calls_before
        if count_invalid_retries:
            self.invalid_responses += max(0, calls_used - 1)
        return result, None

    def _estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Optional[float]:
        if (
            self.input_cost_per_million is None
            or self.output_cost_per_million is None
        ):
            return None
        return round(
            prompt_tokens * self.input_cost_per_million / 1_000_000
            + completion_tokens * self.output_cost_per_million / 1_000_000,
            8,
        )

    @staticmethod
    def _percentage(
        numerator: int,
        denominator: int,
        *,
        empty_value: float = 0.0,
    ) -> float:
        if denominator == 0:
            return empty_value
        return round(numerator / denominator * 100, 2)

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(value.casefold().split())


def load_cases(path: Path) -> List[EvaluationCase]:
    """Load strict evaluation cases and reject duplicate IDs."""
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError("evaluation case file must contain a JSON array")
    cases = [
        EvaluationCase.model_validate_json(json.dumps(item))
        for item in raw_cases
    ]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case IDs must be unique")
    return cases
