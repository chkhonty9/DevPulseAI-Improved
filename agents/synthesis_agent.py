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
    DeveloperProfile,
    EvidenceItem,
    IntelligenceDigest,
    Signal,
    SynthesisDraft,
    format_validation_error,
    validate_contract,
)
from services import GetQHSEClient, LLMClient
from .claims import contains_sentence, drop_sentences
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

    # Priority is ranking, not reasoning: it is computed deterministically
    # from the agents' relevance and risk results (as in the original design),
    # so a small model cannot drop a high-relevance signal from the list.
    RISK_PRIORITY_MULTIPLIER = {
        "CRITICAL": 2.0,
        "HIGH": 1.5,
        "MEDIUM": 1.0,
        "LOW": 0.8,
    }
    MAX_PRIORITY_SIGNALS = 5

    SYNTHESIS_FOCUS_RULES = """- Write for THIS developer: connect items to their technologies and
  interests (see matched_technologies / matched_interests).
- Items with relevance_score >= 70 are the main content of the digest. Never
  describe them as irrelevant or as "no actionable intelligence".
- A digest is about what is useful to the developer, not only about risks. If
  no item has elevated risk, recommend concrete next steps such as evaluating,
  trying, or reading the most relevant items, and say that no risks were found.
- "No action required" is only acceptable when every item has relevance
  below 40 and LOW risk."""

    def synthesize(
        self,
        signals: List[Dict[str, Any]],
        profile: DeveloperProfile | Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Generate and validate one cited intelligence digest."""
        validated_signals = self._validate_signals(signals)
        if not validated_signals:
            raise ContractValidationError(
                "Synthesis requires at least one assessed signal."
            )

        evidence_pack = self.build_evidence_pack(validated_signals)
        evidence_payload = self._evidence_payload(evidence_pack, validated_signals)
        known_ids = {item.evidence_id for item in evidence_pack}
        profile_block = self._profile_block(profile)

        prompt = f"""Synthesize a focused intelligence digest for a developer.

{profile_block}You may use ONLY the evidence pack below. Each item has a stable evidence_id.
Do not invent facts, identifiers, versions, vulnerabilities, or sources.

Focus rules:
{self.SYNTHESIS_FOCUS_RULES}

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
        profile: DeveloperProfile | Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Revise one digest from critic feedback using the same evidence set."""
        if revision_number not in {1, 2}:
            raise ContractValidationError("revision_number must be 1 or 2")

        validated_signals = self._validate_signals(signals)
        evidence_pack = self.build_evidence_pack(validated_signals)
        known_ids = {item.evidence_id for item in evidence_pack}
        evidence_payload = self._evidence_payload(evidence_pack, validated_signals)
        profile_block = self._profile_block(profile)
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
        flagged_sentences = list(critic.unsupported_claims)
        flagged_block = (
            "Sentences the critic flagged as unsupported (exact text from the "
            "previous report). Rewrite each one so it is fully supported by "
            "its cited evidence, or remove it:\n"
            + "\n".join(f"- {sentence}" for sentence in flagged_sentences)
            + "\n\n"
            if flagged_sentences
            else ""
        )
        prompt = f"""Revise this developer intelligence report.

Revision number: {revision_number} of 2.

{profile_block}Previous report (may repeat untrusted external text):
{wrap_untrusted("previous_report", json.dumps(previous_report, ensure_ascii=False, indent=2))}

Independent critic feedback:
{json.dumps(review_feedback, ensure_ascii=False, indent=2)}

Allowed evidence pack (untrusted external data):
{wrap_untrusted("evidence_pack", json.dumps(evidence_payload, ensure_ascii=False, indent=2))}

{flagged_block}Fix the critic's issues without adding unsupported facts. Avoid overstated words
such as "perfectly", "explicitly", "all", or "critical" unless the evidence
says so. Use only evidence IDs
from the evidence pack. Every summary and recommendation claim must remain
traceable to its citations. If the critic's profile_relevance_score is below
70, rewrite the summary and recommendations so they clearly connect the most
relevant items to the developer's technologies and interests; do not return
the same text again.

Focus rules:
{self.SYNTHESIS_FOCUS_RULES}

Return the COMPLETE revised report as one JSON object with exactly these
fields: executive_summary, summary_evidence_ids, priority_evidence_ids, and
recommendations (each item with text and evidence_ids). The report must be
different from the previous one. Do not add other fields.
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
                flagged_sentences,
            ),
            task_name=f"Evidence-grounded synthesis revision {revision_number}",
            model=self.model_id,
            temperature=0.2,
            max_tokens=1400,
        )
        draft = SynthesisDraft.model_validate(draft_payload)
        return self._assemble_digest(validated_signals, evidence_pack, draft)

    def remove_flagged_sentences(
        self,
        signals: List[Dict[str, Any]],
        previous_digest: Mapping[str, Any],
        flagged_sentences: Sequence[str],
    ) -> Dict[str, Any]:
        """Conservative fallback edit: delete flagged sentences, add nothing.

        Used only when the model could not rewrite them. Removing a sentence
        cannot introduce an unsupported fact; the critic still reviews the
        result. Refuses when the summary or all recommendations would vanish.
        """
        if not flagged_sentences:
            raise ContractValidationError("No flagged sentences to remove.")
        validated_signals = self._validate_signals(signals)
        evidence_pack = self.build_evidence_pack(validated_signals)
        previous = self._validate_previous_digest(previous_digest)

        summary = drop_sentences(previous.executive_summary, flagged_sentences)
        recommendations = []
        for item in previous.recommendations:
            text = drop_sentences(item.text, flagged_sentences)
            if text:
                recommendations.append(
                    {"text": text, "evidence_ids": list(item.evidence_ids)}
                )
        if not summary or not recommendations:
            raise ContractValidationError(
                "Removing the flagged sentences would leave the report empty."
            )
        if summary == previous.executive_summary and len(recommendations) == len(
            previous.recommendations
        ) and all(
            new["text"] == old.text
            for new, old in zip(recommendations, previous.recommendations)
        ):
            raise ContractValidationError(
                "None of the flagged sentences were found in the report."
            )

        evidence_by_signal = {
            item.signal_id: item.evidence_id for item in evidence_pack
        }
        draft = SynthesisDraft.model_validate(
            {
                "executive_summary": summary,
                "summary_evidence_ids": list(previous.summary_evidence_ids),
                "priority_evidence_ids": [
                    evidence_by_signal[signal.id]
                    for signal in previous.priority_signals
                    if signal.id in evidence_by_signal
                ][:5]
                or [evidence_pack[0].evidence_id],
                "recommendations": recommendations,
            }
        )
        return self._assemble_digest(validated_signals, evidence_pack, draft)

    @classmethod
    def _validate_revision_patch(
        cls,
        payload: Dict[str, Any],
        previous_report: Mapping[str, Any],
        known_ids: set[str],
        flagged_sentences: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Merge a revision with the last validated report and check it changed.

        A revision that keeps a flagged sentence word for word, or returns the
        same report, gets one correction attempt instead of being accepted.
        """
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
        merged = cls._validate_draft({**previous_report, **payload}, known_ids)

        report_text = " ".join(
            [
                str(merged.get("executive_summary", "")),
                *(
                    str(item.get("text", ""))
                    for item in merged.get("recommendations", [])
                ),
            ]
        )
        still_present = [
            sentence
            for sentence in flagged_sentences
            if contains_sentence(report_text, sentence)
        ]
        if still_present:
            raise AgentOutputError(
                "the revision still contains flagged sentences word for word; "
                "rewrite or remove: " + " | ".join(still_present)
            )

        comparable_fields = ("executive_summary", "recommendations")
        if all(
            merged.get(field) == previous_report.get(field)
            for field in comparable_fields
        ):
            raise AgentOutputError(
                "the revision is identical to the previous report; change the "
                "summary or recommendations to address the critic feedback"
            )
        return merged

    @classmethod
    def _assemble_digest(
        cls,
        validated_signals: List[Dict[str, Any]],
        evidence_pack: Tuple[EvidenceItem, ...],
        draft: SynthesisDraft,
    ) -> Dict[str, Any]:
        priority_signals = cls.rank_priority_signals(validated_signals)
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

    @classmethod
    def rank_priority_signals(
        cls,
        validated_signals: Sequence[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Rank by relevance x risk multiplier; ties keep input order."""

        def priority_score(signal: Mapping[str, Any]) -> float:
            relevance = float(signal["relevance"]["score"])
            risk_level = signal["risk"]["risk_level"]
            return relevance * cls.RISK_PRIORITY_MULTIPLIER.get(risk_level, 1.0)

        ranked = sorted(validated_signals, key=priority_score, reverse=True)
        return list(ranked[: cls.MAX_PRIORITY_SIGNALS])

    @staticmethod
    def _evidence_payload(
        evidence_pack: Sequence[EvidenceItem],
        validated_signals: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Evidence shown to the model, including the profile matches."""
        signals_by_id = {signal["id"]: signal for signal in validated_signals}
        payload = []
        for item in evidence_pack:
            relevance = signals_by_id[item.signal_id]["relevance"]
            payload.append(
                {
                    "evidence_id": item.evidence_id,
                    "claim": item.claim,
                    "relevance_score": item.relevance_score,
                    "matched_technologies": relevance.get(
                        "matched_technologies", []
                    ),
                    "matched_interests": relevance.get("matched_interests", []),
                    "actionability": relevance.get("actionability"),
                    "risk_level": item.risk_level,
                }
            )
        return payload

    @staticmethod
    def _profile_block(
        profile: DeveloperProfile | Mapping[str, Any] | None,
    ) -> str:
        """Render the developer profile for synthesis prompts."""
        if profile is None:
            return ""
        validated = (
            profile
            if isinstance(profile, DeveloperProfile)
            else validate_contract(
                DeveloperProfile,
                profile,
                context="Synthesis developer profile",
            )
        )
        technologies = ", ".join(validated.technologies) or "None supplied"
        interests = ", ".join(validated.interests) or "None supplied"
        return (
            "Developer profile (this digest is written for this person):\n"
            f"- Technologies: {technologies}\n"
            f"- Interests: {interests}\n"
            f"- Experience level: {validated.experience_level}\n\n"
        )

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
