"""Strict domain contracts shared by every DevPulseAI pipeline stage."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Mapping, Optional, Type, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class ContractModel(BaseModel):
    """Base contract: reject unknown fields and implicit type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class DeveloperProfile(ContractModel):
    technologies: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    experience_level: Literal["beginner", "intermediate", "advanced"]

    @field_validator("technologies", "interests")
    @classmethod
    def reject_blank_profile_items(cls, values: List[str]) -> List[str]:
        if any(not value.strip() for value in values):
            raise ValueError("items must be non-empty strings")
        return values

    @model_validator(mode="after")
    def require_personal_context(self) -> "DeveloperProfile":
        if not self.technologies and not self.interests:
            raise ValueError("add at least one technology or interest")
        return self


class RelevanceResult(ContractModel):
    score: float = Field(ge=0, le=100)
    matched_technologies: List[str]
    matched_interests: List[str]
    reasoning: str = Field(min_length=1)
    actionability: Literal["LOW", "MEDIUM", "HIGH"]

    @field_validator("matched_technologies", "matched_interests")
    @classmethod
    def reject_blank_matches(cls, values: List[str]) -> List[str]:
        if any(not value.strip() for value in values):
            raise ValueError("matches must contain non-empty strings")
        return values


class RiskResult(ContractModel):
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    concerns: List[str]
    evidence_quotes: List[str]
    recommended_actions: List[str]
    breaking_changes: bool

    @field_validator("concerns", "evidence_quotes", "recommended_actions")
    @classmethod
    def reject_blank_risk_items(cls, values: List[str]) -> List[str]:
        if any(not value.strip() for value in values):
            raise ValueError("items must contain non-empty strings")
        return values

    @model_validator(mode="after")
    def require_one_to_one_evidence(self) -> "RiskResult":
        item_count = len(self.concerns)
        if len(self.evidence_quotes) != item_count:
            raise ValueError("each concern must have one evidence quote")
        if len(self.recommended_actions) != item_count:
            raise ValueError("each concern must have one recommended action")
        if self.risk_level == "LOW" and item_count != 0:
            raise ValueError("LOW risk must not include speculative concerns")
        if self.risk_level != "LOW" and item_count == 0:
            raise ValueError("non-LOW risk requires at least one evidenced concern")
        if self.breaking_changes and item_count == 0:
            raise ValueError("breaking changes require at least one evidenced concern")
        if self.breaking_changes and self.risk_level == "LOW":
            raise ValueError("breaking changes require at least MEDIUM risk")
        return self


class Signal(ContractModel):
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str
    url: str
    metadata: Dict[str, Any]
    collected_at: datetime
    relevance: Optional[RelevanceResult] = None
    risk: Optional[RiskResult] = None


class EvidenceItem(ContractModel):
    """Immutable source material that synthesis is allowed to cite."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        frozen=True,
    )

    evidence_id: str = Field(pattern=r"^EV-[A-F0-9]{12}$")
    signal_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    source_url: str
    relevance_score: float = Field(ge=0, le=100)
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class CitedRecommendation(ContractModel):
    """An AI recommendation with at least one verifiable citation."""

    text: str = Field(min_length=1)
    evidence_ids: List[str] = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, values: List[str]) -> List[str]:
        if any(not value.strip() for value in values):
            raise ValueError("evidence IDs must be non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError("evidence IDs must not contain duplicates")
        return values


class SynthesisDraft(ContractModel):
    """Strict JSON contract returned directly by the synthesis model."""

    executive_summary: str = Field(min_length=1)
    summary_evidence_ids: List[str] = Field(min_length=1)
    priority_evidence_ids: List[str] = Field(min_length=1, max_length=5)
    recommendations: List[CitedRecommendation] = Field(min_length=1)

    @field_validator("summary_evidence_ids", "priority_evidence_ids")
    @classmethod
    def validate_citation_lists(cls, values: List[str]) -> List[str]:
        if any(not value.strip() for value in values):
            raise ValueError("evidence IDs must be non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError("evidence IDs must not contain duplicates")
        return values


class IntelligenceDigest(ContractModel):
    generated_at: datetime
    total_signals: int = Field(ge=0)
    executive_summary: str = Field(min_length=1)
    summary_evidence_ids: List[str] = Field(min_length=1)
    priority_signals: List[Signal]
    signals_by_source: Dict[str, List[Signal]]
    recommendations: List[CitedRecommendation] = Field(min_length=1)
    evidence_pack: List[EvidenceItem] = Field(min_length=1)

    @model_validator(mode="after")
    def require_traceable_citations(self) -> "IntelligenceDigest":
        evidence_ids = [item.evidence_id for item in self.evidence_pack]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence pack IDs must be unique")

        known_ids = set(evidence_ids)
        cited_ids = set(self.summary_evidence_ids)
        for recommendation in self.recommendations:
            cited_ids.update(recommendation.evidence_ids)
        unknown_ids = sorted(cited_ids - known_ids)
        if unknown_ids:
            raise ValueError(
                "digest cites unknown evidence IDs: " + ", ".join(unknown_ids)
            )

        evidence_signal_ids = {item.signal_id for item in self.evidence_pack}
        unknown_priority_signals = sorted(
            signal.id
            for signal in self.priority_signals
            if signal.id not in evidence_signal_ids
        )
        if unknown_priority_signals:
            raise ValueError(
                "priority signals are missing from the evidence pack: "
                + ", ".join(unknown_priority_signals)
            )
        return self


class CriticResult(ContractModel):
    passed: bool
    groundedness_score: float = Field(ge=0, le=100)
    profile_relevance_score: float = Field(ge=0, le=100)
    completeness_score: float = Field(ge=0, le=100)
    unsupported_claims: List[str]
    feedback: List[str]

    @field_validator("unsupported_claims", "feedback")
    @classmethod
    def reject_blank_critic_items(cls, values: List[str]) -> List[str]:
        if any(not value.strip() for value in values):
            raise ValueError("critic items must be non-empty strings")
        return values

    @model_validator(mode="after")
    def require_passing_scores(self) -> "CriticResult":
        scores = (
            self.groundedness_score,
            self.profile_relevance_score,
            self.completeness_score,
        )
        if self.passed and (min(scores) < 70 or self.unsupported_claims):
            raise ValueError(
                "a passing review requires all scores >= 70 and no "
                "unsupported claims"
            )
        return self


class CriticDraft(ContractModel):
    """Strict model output before deterministic pass/fail calculation."""

    groundedness_score: float = Field(ge=0, le=100)
    profile_relevance_score: float = Field(ge=0, le=100)
    completeness_score: float = Field(ge=0, le=100)
    unsupported_claims: List[str]
    feedback: List[str]

    @field_validator("unsupported_claims", "feedback")
    @classmethod
    def reject_blank_draft_items(cls, values: List[str]) -> List[str]:
        if any(not value.strip() for value in values):
            raise ValueError("critic items must be non-empty strings")
        return values


class RevisionRecord(ContractModel):
    """One critic verdict in the bounded synthesis revision history."""

    revision_number: int = Field(ge=0, le=2)
    digest_summary: str = Field(min_length=1)
    critic: CriticResult


class PipelineResult(ContractModel):
    status: Literal["completed", "needs_review"]
    signals: List[Signal]
    digest: IntelligenceDigest
    critic: Optional[CriticResult] = None
    evidence: List[EvidenceItem] = Field(default_factory=list)
    revision_history: List[RevisionRecord] = Field(default_factory=list)
    revision_attempts: int = Field(default=0, ge=0, le=2)
    revision_failure: Optional[str] = None

    @model_validator(mode="after")
    def require_status_to_match_critic(self) -> "PipelineResult":
        if self.critic is None:
            raise ValueError("pipeline result requires a final critic verdict")
        if self.status == "completed" and not self.critic.passed:
            raise ValueError("completed status requires a passing critic")
        if self.status == "needs_review" and self.critic.passed:
            raise ValueError("needs_review status requires a failing critic")
        if self.revision_history:
            expected_numbers = list(range(len(self.revision_history)))
            actual_numbers = [
                record.revision_number for record in self.revision_history
            ]
            if actual_numbers != expected_numbers:
                raise ValueError("revision history numbers must be sequential")
        return self


class ContractValidationError(ValueError):
    """A pipeline boundary received data that violates its public contract."""


ModelType = TypeVar("ModelType", bound=ContractModel)


def format_validation_error(exc: ValidationError) -> str:
    """Return compact, user-safe field errors without dumping input values."""
    issues = []
    for error in exc.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in error["loc"]) or "value"
        issues.append(f"{location}: {error['msg']}")
    return "; ".join(issues)


def validate_contract(
    model_type: Type[ModelType],
    payload: Mapping[str, Any],
    *,
    context: str,
) -> ModelType:
    """Validate a mapping and raise a concise visible pipeline error."""
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise ContractValidationError(
            f"{context} failed validation: {format_validation_error(exc)}"
        ) from exc

