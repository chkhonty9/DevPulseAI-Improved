"""Independent quality review for evidence-grounded intelligence digests."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional

from pydantic import ValidationError

from config import DEFAULT_GETQHSE_MODEL, resolve_getqhse_settings
from models import (
    ContractValidationError,
    CriticDraft,
    CriticResult,
    DeveloperProfile,
    IntelligenceDigest,
    format_validation_error,
    validate_contract,
)
from services import GetQHSEClient, LLMClient
from .reasoning import AgentOutputError, complete_validated_json


DEFAULT_MODEL = os.environ.get(
    "MODEL_CRITIC",
    os.environ.get("GETQHSE_MODEL", DEFAULT_GETQHSE_MODEL),
)


class CriticAgent:
    """Review a digest with deterministic integrity checks plus AI scoring."""

    PASSING_SCORE = 70.0

    def __init__(
        self,
        model_id: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.model_id = model_id or DEFAULT_MODEL
        self.llm_client = llm_client

    def _get_llm_client(self) -> LLMClient:
        if self.llm_client is None:
            settings = resolve_getqhse_settings(model=self.model_id)
            self.llm_client = GetQHSEClient(settings)
        return self.llm_client

    def review(
        self,
        digest: Mapping[str, Any],
        profile: DeveloperProfile | Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return a strict review; unknown IDs fail before the AI call."""
        deterministic_issues = self._deterministic_id_checks(digest)
        if deterministic_issues:
            return CriticResult(
                passed=False,
                groundedness_score=0.0,
                profile_relevance_score=0.0,
                completeness_score=0.0,
                unsupported_claims=deterministic_issues,
                feedback=[
                    "Repair the report identifiers before requesting an AI review."
                ],
            ).model_dump(mode="json")

        validated_digest = self._validate_digest(digest)
        if isinstance(profile, DeveloperProfile):
            validated_profile = profile
        else:
            validated_profile = validate_contract(
                DeveloperProfile,
                profile,
                context="Critic developer profile",
            )

        review_packet = {
            "developer_profile": validated_profile.model_dump(mode="json"),
            "executive_summary": validated_digest.executive_summary,
            "summary_evidence_ids": validated_digest.summary_evidence_ids,
            "recommendations": [
                recommendation.model_dump(mode="json")
                for recommendation in validated_digest.recommendations
            ],
            "priority_signals": [
                {
                    "signal_id": signal.id,
                    "title": signal.title,
                    "relevance": signal.relevance.model_dump(mode="json")
                    if signal.relevance
                    else None,
                    "risk": signal.risk.model_dump(mode="json")
                    if signal.risk
                    else None,
                }
                for signal in validated_digest.priority_signals
            ],
            "evidence_pack": [
                item.model_dump(mode="json")
                for item in validated_digest.evidence_pack
            ],
        }
        prompt = f"""Independently review this developer intelligence report.

Report packet:
{json.dumps(review_packet, ensure_ascii=False, indent=2)}

Score from 0 to 100:
- groundedness_score: summary and recommendations supported by cited evidence.
- profile_relevance_score: priorities and actions useful for the profile.
- completeness_score: citations, priorities, and actionable guidance are complete.

List each unsupported factual claim verbatim. Use an empty list when none.
Give short, specific feedback. Do not add fields.

Return only this JSON structure:
{{"groundedness_score":0,"profile_relevance_score":0,"completeness_score":0,"unsupported_claims":[],"feedback":[]}}
"""
        draft_payload = complete_validated_json(
            llm_client=self._get_llm_client(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an independent evidence and relevance critic. "
                        "Return only the requested JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            validator=self._validate_draft,
            task_name="Independent digest review",
            model=self.model_id,
            temperature=0.0,
            max_tokens=900,
        )
        draft = CriticDraft.model_validate(draft_payload)
        passed = (
            min(
                draft.groundedness_score,
                draft.profile_relevance_score,
                draft.completeness_score,
            )
            >= self.PASSING_SCORE
            and not draft.unsupported_claims
        )
        return CriticResult(
            passed=passed,
            **draft.model_dump(mode="python"),
        ).model_dump(mode="json")

    @staticmethod
    def _validate_draft(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            draft = CriticDraft.model_validate(payload)
        except ValidationError as exc:
            raise AgentOutputError(format_validation_error(exc)) from exc
        return draft.model_dump(mode="json")

    @staticmethod
    def _validate_digest(digest: Mapping[str, Any]) -> IntelligenceDigest:
        """Validate a digest through JSON mode so ISO datetimes remain valid."""
        try:
            return IntelligenceDigest.model_validate_json(
                json.dumps(
                    digest,
                    default=lambda value: value.isoformat()
                    if hasattr(value, "isoformat")
                    else str(value),
                )
            )
        except ValidationError as exc:
            raise ContractValidationError(
                "Critic input digest failed validation: "
                + format_validation_error(exc)
            ) from exc

    @staticmethod
    def _deterministic_id_checks(digest: Mapping[str, Any]) -> list[str]:
        """Find unknown/duplicate evidence and signal IDs without AI judgment."""
        issues: list[str] = []
        evidence_pack = digest.get("evidence_pack", [])
        if not isinstance(evidence_pack, list):
            return ["evidence_pack is not a list"]

        evidence_ids = [
            item.get("evidence_id")
            for item in evidence_pack
            if isinstance(item, Mapping)
        ]
        known_evidence_ids = {
            value for value in evidence_ids if isinstance(value, str)
        }
        duplicate_evidence_ids = sorted(
            {
                value
                for value in known_evidence_ids
                if evidence_ids.count(value) > 1
            }
        )
        if duplicate_evidence_ids:
            issues.append(
                "Duplicate evidence IDs: " + ", ".join(duplicate_evidence_ids)
            )

        cited_ids = set()
        summary_ids = digest.get("summary_evidence_ids", [])
        if isinstance(summary_ids, list):
            cited_ids.update(value for value in summary_ids if isinstance(value, str))
        recommendations = digest.get("recommendations", [])
        if isinstance(recommendations, list):
            for recommendation in recommendations:
                if not isinstance(recommendation, Mapping):
                    continue
                ids = recommendation.get("evidence_ids", [])
                if isinstance(ids, list):
                    cited_ids.update(value for value in ids if isinstance(value, str))
        unknown_evidence_ids = sorted(cited_ids - known_evidence_ids)
        if unknown_evidence_ids:
            issues.append(
                "Unknown evidence IDs: " + ", ".join(unknown_evidence_ids)
            )

        grouped_signal_ids = set()
        grouped = digest.get("signals_by_source", {})
        if isinstance(grouped, Mapping):
            for source_signals in grouped.values():
                if not isinstance(source_signals, list):
                    continue
                for signal in source_signals:
                    if isinstance(signal, Mapping) and isinstance(signal.get("id"), str):
                        grouped_signal_ids.add(signal["id"])

        evidence_signal_ids = {
            item.get("signal_id")
            for item in evidence_pack
            if isinstance(item, Mapping) and isinstance(item.get("signal_id"), str)
        }
        unknown_evidence_signals = sorted(evidence_signal_ids - grouped_signal_ids)
        if unknown_evidence_signals:
            issues.append(
                "Evidence references unknown signal IDs: "
                + ", ".join(unknown_evidence_signals)
            )

        priority_signals = digest.get("priority_signals", [])
        priority_signal_ids = (
            {
                signal.get("id")
                for signal in priority_signals
                if isinstance(signal, Mapping)
                and isinstance(signal.get("id"), str)
            }
            if isinstance(priority_signals, list)
            else set()
        )
        unknown_priority_signals = sorted(priority_signal_ids - grouped_signal_ids)
        if unknown_priority_signals:
            issues.append(
                "Unknown priority signal IDs: "
                + ", ".join(unknown_priority_signals)
            )
        return issues
