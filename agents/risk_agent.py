"""AI risk analysis for technical signals."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import ValidationError

from config import DEFAULT_GETQHSE_MODEL, resolve_getqhse_settings
from models import RiskResult, Signal, format_validation_error, validate_contract
from services import GetQHSEClient, LLMClient
from .reasoning import AgentOutputError, complete_validated_json


DEFAULT_MODEL = os.environ.get(
    "MODEL_RISK",
    os.environ.get("GETQHSE_MODEL", DEFAULT_GETQHSE_MODEL),
)


class RiskAgent:
    """Assess technical risk using the selected AI model client."""

    RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    SENSITIVE_IDENTIFIER_PATTERNS = (
        re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE),
        re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:[-+][A-Za-z0-9.-]+)?\b"),
    )
    MIGRATION_REQUIREMENT_PATTERN = re.compile(
        r"\b(?:must|required|mandatory)\b.{0,40}\b(?:migrat\w*|upgrad\w*)\b",
        re.IGNORECASE,
    )

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

    def assess(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Return a validated AI assessment or raise a visible error."""
        validated_signal = validate_contract(
            Signal,
            signal,
            context="Risk agent input signal",
        )
        signal = validated_signal.model_dump(mode="python", exclude_none=True)
        prompt = f"""Analyze only the explicit risk facts in this source text.

Source: {signal.get('source', 'unknown')}
Title: {signal.get('title', 'Untitled')}
Description: {signal.get('description', '')[:500]}

Return exactly five keys: risk_level, concerns, evidence_quotes,
recommended_actions, breaking_changes. Do not add any other key.

If the title and description do not explicitly state a vulnerability,
breaking or incompatible change, removal, or deprecation, return exactly:
{{"risk_level":"LOW","concerns":[],"evidence_quotes":[],"recommended_actions":[],"breaking_changes":false}}

Otherwise, use LOW, MEDIUM, HIGH, or CRITICAL for risk_level. Every concern
must have one matching evidence quote copied from the source text and one
conservative action. All three fields are arrays of plain JSON strings, never
objects. Do not invent identifiers, versions, or requirements.

Return one JSON object only."""

        return complete_validated_json(
            llm_client=self._get_llm_client(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Assess only explicit technical risk facts. The only "
                        "allowed top-level keys are risk_level, concerns, "
                        "evidence_quotes, recommended_actions, and "
                        "breaking_changes. Return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            validator=lambda payload: self._validate_response(payload, signal),
            task_name=(
                "Risk analysis for "
                f"'{str(signal.get('title', 'Untitled'))[:80]}'"
            ),
            model=self.model_id,
            temperature=0.0,
            max_tokens=650,
        )

    def assess_batch(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assess multiple signals and stop immediately on an AI failure."""
        return [{**signal, "risk": self.assess(signal)} for signal in signals]

    @classmethod
    def _validate_response(
        cls,
        payload: Dict[str, Any],
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = cls._normalize_unambiguous_string_wrappers(payload)
        try:
            result = RiskResult.model_validate(payload)
        except ValidationError as exc:
            raise AgentOutputError(format_validation_error(exc)) from exc

        source_text = " ".join(
            [
                str(signal.get("title", "")),
                str(signal.get("description", "")),
            ]
        )
        normalized_source = cls._normalize_text(source_text)
        for quote in result.evidence_quotes:
            if cls._normalize_text(quote) not in normalized_source:
                raise AgentOutputError(
                    "every evidence quote must appear exactly in the signal text"
                )

        generated_claims = " ".join(
            [*result.concerns, *result.recommended_actions]
        )
        for pattern in cls.SENSITIVE_IDENTIFIER_PATTERNS:
            for identifier in pattern.findall(generated_claims):
                if identifier.casefold() not in source_text.casefold():
                    raise AgentOutputError(
                        f"unsupported identifier or version: {identifier}"
                    )

        if (
            cls.MIGRATION_REQUIREMENT_PATTERN.search(generated_claims)
            and not cls.MIGRATION_REQUIREMENT_PATTERN.search(source_text)
        ):
            raise AgentOutputError(
                "migration or upgrade requirements must be explicit in the source"
            )

        if result.breaking_changes:
            breaking_markers = (
                "breaking",
                "incompatible",
                "deprecat",
                "remove",
                "no longer",
            )
            if not any(marker in normalized_source for marker in breaking_markers):
                raise AgentOutputError(
                    "breaking_changes cannot be true without explicit source evidence"
                )
        return result.model_dump(mode="json")

    @classmethod
    def _normalize_unambiguous_string_wrappers(
        cls,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Unwrap common model-produced objects without inventing content.

        Small models sometimes return ``[{"text": "..."}]`` where the public
        contract requires ``["..."]``. We unwrap only a single recognized,
        non-empty string value. Ambiguous objects are left untouched so strict
        Pydantic validation still rejects them and triggers one correction.
        """
        normalized = dict(payload)

        # Canonicalize the model's own LOW/no-breaking decision. Small models
        # sometimes append speculative concerns or category fields even after
        # selecting LOW. Removing those unsupported additions is safer than
        # displaying them and preserves the model's actual classification.
        if (
            normalized.get("risk_level") == "LOW"
            and normalized.get("breaking_changes") is False
        ):
            return {
                "risk_level": "LOW",
                "concerns": [],
                "evidence_quotes": [],
                "recommended_actions": [],
                "breaking_changes": False,
            }

        field_aliases = {
            "concerns": ("concern", "text", "description", "issue", "risk"),
            "evidence_quotes": (
                "evidence_quote",
                "quote",
                "evidence",
                "text",
            ),
            "recommended_actions": (
                "recommended_action",
                "action",
                "recommendation",
                "text",
            ),
        }

        for field_name, aliases in field_aliases.items():
            values = normalized.get(field_name)
            if not isinstance(values, list):
                continue
            normalized[field_name] = [
                cls._unwrap_string_item(value, aliases) for value in values
            ]
        return normalized

    @staticmethod
    def _unwrap_string_item(
        value: Any,
        aliases: Sequence[str],
    ) -> Any:
        if not isinstance(value, dict):
            return value

        candidates = {
            candidate.strip()
            for alias in aliases
            if isinstance((candidate := value.get(alias)), str)
            and candidate.strip()
        }
        if len(candidates) == 1:
            return candidates.pop()
        return value

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(value.casefold().split())
