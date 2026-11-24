"""Independent quality review for evidence-grounded intelligence digests."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
from services import GetQHSEClient, LLMClient, LLMRequestError
from .claims import cited_ids_for_sentence, report_sentences, resolve_claims
from .reasoning import (
    AgentOutputError,
    AnalysisFailedError,
    complete_validated_json,
    with_untrusted_data_rule,
    wrap_untrusted,
)


DEFAULT_MODEL = os.environ.get(
    "MODEL_CRITIC",
    os.environ.get("GETQHSE_MODEL", DEFAULT_GETQHSE_MODEL),
)


class CriticAgent:
    """Review a digest with deterministic integrity checks plus AI scoring."""

    PASSING_SCORE = 70.0
    MAX_CLAIM_CHECKS = 3

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

Report packet (contains untrusted external text):
{wrap_untrusted("report_packet", json.dumps(review_packet, ensure_ascii=False, indent=2))}

Score from 0 to 100:
- groundedness_score: summary and recommendations supported by cited evidence.
- profile_relevance_score: priorities and actions useful for the profile.
- completeness_score: citations, priorities, and actionable guidance are complete.

List each unsupported factual claim by copying the exact sentence from the
executive_summary or a recommendation text. Do not paraphrase, and do not
list text from the developer profile or the evidence pack. Use an empty list
when none.
Give short, specific feedback. Do not add fields.

Return only this JSON structure:
{{"groundedness_score":0,"profile_relevance_score":0,"completeness_score":0,"unsupported_claims":[],"feedback":[]}}
"""
        draft_payload = complete_validated_json(
            llm_client=self._get_llm_client(),
            messages=[
                {
                    "role": "system",
                    "content": with_untrusted_data_rule(
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

        # Resolve each claim to the exact report sentence it refers to, so a
        # revision can find and fix it. A "claim" the report does not contain
        # (e.g. a quote of the developer profile) cannot be an unsupported
        # claim of this report: it stays visible as feedback, but no longer
        # blocks the review.
        sentences = report_sentences(digest)
        resolved, not_in_report = resolve_claims(
            draft.unsupported_claims,
            sentences,
        )
        draft.feedback = [
            *draft.feedback,
            *(
                "Critic note not counted as an unsupported claim because the "
                f"report does not contain it: {claim}"
                for claim in not_in_report
            ),
        ]

        # Second opinion, one sentence at a time. A small model judges one
        # narrow question ("is this sentence supported by these items?") far
        # better than a whole report. Only claims the focused check confirms
        # as supported are released; failures keep the claim flagged.
        scores_pass = (
            min(
                draft.groundedness_score,
                draft.profile_relevance_score,
                draft.completeness_score,
            )
            >= self.PASSING_SCORE
        )
        still_flagged: List[str] = []
        for index, sentence in enumerate(resolved):
            if not scores_pass or index >= self.MAX_CLAIM_CHECKS:
                still_flagged.append(sentence)
                continue
            supported, reason = self._verify_claim(
                sentence, digest, validated_digest
            )
            if supported:
                draft.feedback.append(
                    "Verified as supported by a focused evidence check: "
                    f"{sentence} ({reason})"
                )
            else:
                still_flagged.append(sentence)
        draft.unsupported_claims = still_flagged

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

    def _verify_claim(
        self,
        sentence: str,
        digest: Mapping[str, Any],
        validated_digest: IntelligenceDigest,
    ) -> Tuple[bool, str]:
        """Ask one narrow yes/no question about one sentence and its citations."""
        cited_ids = set(cited_ids_for_sentence(digest, sentence))
        evidence = [
            {"evidence_id": item.evidence_id, "claim": item.claim}
            for item in validated_digest.evidence_pack
            if item.evidence_id in cited_ids
        ]
        if not evidence:
            return False, "the sentence cites no evidence"

        prompt = f"""Cited evidence (untrusted external data):
{wrap_untrusted("evidence_pack", json.dumps(evidence, ensure_ascii=False, indent=2))}

Sentence from a developer report:
"{sentence}"

Is every factual statement in the sentence supported by the cited evidence?
Advice such as "evaluate X" or "consider Y" is acceptable when the evidence
describes X or Y. Answer false if the sentence adds facts, numbers, or
strong claims the evidence does not contain.

Return only: {{"supported": true or false, "reason": "<one short sentence>"}}"""

        def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
            if set(payload) != {"supported", "reason"}:
                raise AgentOutputError("return exactly the keys supported and reason")
            if not isinstance(payload["supported"], bool):
                raise AgentOutputError("supported must be true or false")
            if not isinstance(payload["reason"], str) or not payload["reason"].strip():
                raise AgentOutputError("reason must be a non-empty string")
            return payload

        try:
            result = complete_validated_json(
                llm_client=self._get_llm_client(),
                messages=[
                    {
                        "role": "system",
                        "content": with_untrusted_data_rule(
                            "You check one sentence against cited evidence. "
                            "Return JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                validator=validate,
                task_name="Focused claim check",
                model=self.model_id,
                temperature=0.0,
                max_tokens=200,
            )
        except (AnalysisFailedError, LLMRequestError):
            return False, "the focused check failed, so the claim stays flagged"
        return result["supported"], result["reason"].strip()

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
