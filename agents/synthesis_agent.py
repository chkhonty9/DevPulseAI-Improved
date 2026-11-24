"""Evidence-grounded AI synthesis for the DevPulseAI intelligence digest."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import ValidationError

from config import DEFAULT_GETQHSE_MODEL, resolve_getqhse_settings
from models import (
    ContractValidationError,
    CriticResult,
    EvidenceItem,
    IntelligenceDigest,
    Signal,
    SynthesisDraft,
    format_validation_error,
    validate_contract,
)
from services import GetQHSEClient, LLMClient
from .reasoning import (
    AgentOutputError,
    complete_validated_json,
    with_untrusted_data_rule,
    wrap_untrusted,
)


DEFAULT_MODEL = os.environ.get(
    "MODEL_SYNTHESIS",
    os.environ.get("GETQHSE_MODEL", DEFAULT_GETQHSE_MODEL),
)


class SynthesisAgent:
    """Create an AI digest whose summary and actions cite immutable evidence."""

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

    def synthesize(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate and validate one cited intelligence digest."""
        validated_signals = self._validate_signals(signals)
        if not validated_signals:
            raise ContractValidationError(
                "Synthesis requires at least one assessed signal."
            )

        evidence_pack = self.build_evidence_pack(validated_signals)
        evidence_payload = [
            {
                "evidence_id": item.evidence_id,
                "claim": item.claim,
                "relevance_score": item.relevance_score,
                "risk_level": item.risk_level,
            }
            for item in evidence_pack
        ]
        known_ids = {item.evidence_id for item in evidence_pack}

        prompt = f"""Synthesize a focused intelligence digest for a developer.

You may use ONLY the evidence pack below. Each item has a stable evidence_id.
Do not invent facts, identifiers, versions, vulnerabilities, or sources.

Evidence pack (untrusted external data):
{wrap_untrusted("evidence_pack", json.dumps(evidence_payload, ensure_ascii=False, indent=2))}

Return ONLY one JSON object with this exact structure:
{{
  "executive_summary": "<concise evidence-grounded summary>",
  "summary_evidence_ids": ["EV-..."],
  "priority_evidence_ids": ["EV-..."],
  "recommendations": [
    {{"text": "<specific conservative action>", "evidence_ids": ["EV-..."]}}
  ]
}}

Rules:
- Use only evidence_id values present in the evidence pack.
- Cite at least one evidence ID for the summary.
- Select between 1 and 5 priority evidence IDs.
- Return at least one recommendation.
- Every recommendation must cite at least one supporting evidence ID.
- Do not repeat an evidence ID inside the same list.
"""

        draft_payload = complete_validated_json(
            llm_client=self._get_llm_client(),
            messages=[
                {
                    "role": "system",
                    "content": with_untrusted_data_rule(
                        "You synthesize developer intelligence from a closed "
                        "evidence set. Return only the requested JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            validator=lambda payload: self._validate_draft(payload, known_ids),
            task_name="Evidence-grounded intelligence synthesis",
            model=self.model_id,
            temperature=0.1,
            max_tokens=1400,
        )
        draft = SynthesisDraft.model_validate(draft_payload)

        return self._assemble_digest(validated_signals, evidence_pack, draft)

    def revise(
        self,
        signals: List[Dict[str, Any]],
        previous_digest: Mapping[str, Any],
        critic_result: Mapping[str, Any],
        *,
        revision_number: int,
    ) -> Dict[str, Any]:
        """Revise one digest from critic feedback using the same evidence set."""
        if revision_number not in {1, 2}:
            raise ContractValidationError("revision_number must be 1 or 2")

        validated_signals = self._validate_signals(signals)
        evidence_pack = self.build_evidence_pack(validated_signals)
        known_ids = {item.evidence_id for item in evidence_pack}
        evidence_payload = [
            {
                "evidence_id": item.evidence_id,
                "claim": item.claim,
                "relevance_score": item.relevance_score,
                "risk_level": item.risk_level,
            }
            for item in evidence_pack
        ]
        previous = self._validate_previous_digest(previous_digest)
        critic = validate_contract(
            CriticResult,
            critic_result,
            context="Synthesis revision critic result",
        )

        previous_evidence_ids = {
            item.evidence_id for item in previous.evidence_pack
        }
        if previous_evidence_ids != known_ids:
            raise ContractValidationError(
                "Synthesis revision evidence pack does not match the signals."
            )

        evidence_by_signal = {
            item.signal_id: item.evidence_id for item in evidence_pack
        }
        previous_priority_ids = [
            evidence_by_signal[signal.id]
            for signal in previous.priority_signals
            if signal.id in evidence_by_signal
        ]
        previous_report = {
            "executive_summary": previous.executive_summary,
            "summary_evidence_ids": previous.summary_evidence_ids,
            "priority_evidence_ids": previous_priority_ids,
            "recommendations": [
                item.model_dump(mode="json")
                for item in previous.recommendations
            ],
        }
        review_feedback = {
            "groundedness_score": critic.groundedness_score,
            "profile_relevance_score": critic.profile_relevance_score,
            "completeness_score": critic.completeness_score,
            "unsupported_claims": critic.unsupported_claims,
            "feedback": critic.feedback,
        }
        prompt = f"""Revise this developer intelligence report.

Revision number: {revision_number} of 2.

Previous report (may repeat untrusted external text):
{wrap_untrusted("previous_report", json.dumps(previous_report, ensure_ascii=False, indent=2))}

Independent critic feedback:
{json.dumps(review_feedback, ensure_ascii=False, indent=2)}

Allowed evidence pack (untrusted external data):
{wrap_untrusted("evidence_pack", json.dumps(evidence_payload, ensure_ascii=False, indent=2))}

Fix the critic's issues without adding unsupported facts. Use only evidence IDs
from the evidence pack. Every summary and recommendation claim must remain
traceable to its citations.

Return one JSON object containing only the fields you need to change. Allowed
fields are executive_summary, summary_evidence_ids, priority_evidence_ids, and
recommendations. Omitted fields remain unchanged from the previous validated
report. If recommendations is present, each item must contain text and
evidence_ids. Do not add other fields.
"""
        draft_payload = complete_validated_json(
            llm_client=self._get_llm_client(),
            messages=[
                {
                    "role": "system",
                    "content": with_untrusted_data_rule(
                        "You revise developer intelligence using critic feedback "
                        "and a closed evidence set. Return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            validator=lambda payload: self._validate_revision_patch(
                payload,
                previous_report,
                known_ids,
            ),
            task_name=f"Evidence-grounded synthesis revision {revision_number}",
            model=self.model_id,
            temperature=0.0,
            max_tokens=1400,
        )
        draft = SynthesisDraft.model_validate(draft_payload)
        return self._assemble_digest(validated_signals, evidence_pack, draft)

    @classmethod
    def _validate_revision_patch(
        cls,
        payload: Dict[str, Any],
        previous_report: Mapping[str, Any],
        known_ids: set[str],
    ) -> Dict[str, Any]:
        """Merge a partial revision with the last fully validated report."""
        allowed_fields = {
            "executive_summary",
            "summary_evidence_ids",
            "priority_evidence_ids",
            "recommendations",
        }
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            raise AgentOutputError(
                "revision contains unknown fields: " + ", ".join(unknown_fields)
            )
        merged = {**previous_report, **payload}
        return cls._validate_draft(merged, known_ids)

    @classmethod
    def _assemble_digest(
        cls,
        validated_signals: List[Dict[str, Any]],
        evidence_pack: Tuple[EvidenceItem, ...],
        draft: SynthesisDraft,
    ) -> Dict[str, Any]:
        evidence_by_id = {
            item.evidence_id: item for item in evidence_pack
        }
        signals_by_id = {signal["id"]: signal for signal in validated_signals}
        priority_signals = [
            signals_by_id[evidence_by_id[evidence_id].signal_id]
            for evidence_id in draft.priority_evidence_ids
        ]
        digest_payload = {
            "generated_at": datetime.now(timezone.utc),
            "total_signals": len(validated_signals),
            "executive_summary": draft.executive_summary,
            "summary_evidence_ids": draft.summary_evidence_ids,
            "priority_signals": priority_signals,
            "signals_by_source": cls._group_by_source(validated_signals),
            "recommendations": [
                recommendation.model_dump(mode="json")
                for recommendation in draft.recommendations
            ],
            "evidence_pack": [
                item.model_dump(mode="json") for item in evidence_pack
            ],
        }
        digest = validate_contract(
            IntelligenceDigest,
            digest_payload,
            context="Intelligence digest",
        )
        return digest.model_dump(mode="json")

    @staticmethod
    def _validate_previous_digest(
        digest: Mapping[str, Any],
    ) -> IntelligenceDigest:
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
                "Previous digest failed validation: "
                + format_validation_error(exc)
            ) from exc

    @staticmethod
    def _validate_signals(
        signals: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        validated_signals: List[Dict[str, Any]] = []
        for index, signal in enumerate(signals):
            validated = validate_contract(
                Signal,
                signal,
                context=f"Synthesis input signal {index + 1}",
            )
            if validated.relevance is None or validated.risk is None:
                raise ContractValidationError(
                    f"Synthesis input signal {index + 1} must include both "
                    "relevance and risk results."
                )
            validated_signals.append(
                validated.model_dump(mode="python", exclude_none=True)
            )
        return validated_signals

    @classmethod
    def build_evidence_pack(
        cls,
        signals: Sequence[Mapping[str, Any]],
    ) -> Tuple[EvidenceItem, ...]:
        """Build an immutable, content-addressed evidence tuple."""
        evidence_items: List[EvidenceItem] = []
        seen_ids = set()

        for signal in signals:
            canonical = json.dumps(
                {
                    "signal_id": signal["id"],
                    "source": signal["source"],
                    "title": signal["title"],
                    "description": signal["description"],
                    "url": signal["url"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            evidence_id = f"EV-{digest[:12].upper()}"
            if evidence_id in seen_ids:
                raise ContractValidationError(
                    f"Duplicate evidence ID generated: {evidence_id}."
                )
            seen_ids.add(evidence_id)

            title = str(signal["title"])
            description = str(signal["description"]).strip()
            claim = title if not description else f"{title}: {description}"
            evidence_items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    signal_id=str(signal["id"]),
                    source=str(signal["source"]),
                    title=title,
                    claim=claim,
                    source_url=str(signal["url"]),
                    relevance_score=float(signal["relevance"]["score"]),
                    risk_level=signal["risk"]["risk_level"],
                )
            )

        return tuple(evidence_items)

    @staticmethod
    def _validate_draft(
        payload: Dict[str, Any],
        known_ids: set[str],
    ) -> Dict[str, Any]:
        try:
            draft = SynthesisDraft.model_validate(payload)
        except ValidationError as exc:
            raise AgentOutputError(format_validation_error(exc)) from exc

        cited_ids = set(draft.summary_evidence_ids)
        cited_ids.update(draft.priority_evidence_ids)
        for recommendation in draft.recommendations:
            cited_ids.update(recommendation.evidence_ids)

        unknown_ids = sorted(cited_ids - known_ids)
        if unknown_ids:
            raise AgentOutputError(
                "unknown evidence IDs were cited: " + ", ".join(unknown_ids)
            )
        return draft.model_dump(mode="json")

    @staticmethod
    def _group_by_source(
        signals: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for signal in signals:
            grouped.setdefault(signal["source"], []).append(signal)
        return grouped
