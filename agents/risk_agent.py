"""AI risk analysis for technical signals."""

from __future__ import annotations

import json
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
    # Explicit risk wording. A LOW verdict on text containing one of these
    # terms gets one second look; the model still makes the final decision.
    EXPLICIT_RISK_TERMS = re.compile(
        r"\b(?:remov(?:e|es|ed|al)|deprecat\w*|breaking|incompatib\w*|"
        r"vulnerab\w*|exploit\w*|CVE-\d{4}-\d+|privilege escalation|"
        r"authentication bypass|no longer supported|drops? support|"
        r"end[- ]of[- ]life|requires? migration|must migrate)\b",
        re.IGNORECASE,
    )

    SEVERITY_RUBRIC = """Severity rubric:
- CRITICAL: the text calls a vulnerability critical or actively exploited.
- HIGH: a high-severity vulnerability, or a change that breaks existing
  deployments, data, or authentication (incompatible formats, removed modes).
- MEDIUM: a deprecation, or a removed or changed API that needs code changes.
- LOW: no explicit vulnerability, breaking change, removal, or deprecation."""

    FEW_SHOT_EXAMPLES = """Examples (illustrative, not part of the source):
Text: "ChartKit adds dark mode and faster rendering."
JSON: {"risk_level":"LOW","concerns":[],"evidence_quotes":[],"recommended_actions":[],"breaking_changes":false}
Text: "The CSV importer deprecates the delimiter option; it will be removed later."
JSON: {"risk_level":"MEDIUM","concerns":["The delimiter option is deprecated"],"evidence_quotes":["deprecates the delimiter option"],"recommended_actions":["Find usages of the delimiter option and plan a replacement"],"breaking_changes":true}
Text: "The new cache format is incompatible with existing cache files."
JSON: {"risk_level":"HIGH","concerns":["Existing cache files cannot be read"],"evidence_quotes":["incompatible with existing cache files"],"recommended_actions":["Test the new format against existing cache files before rollout"],"breaking_changes":true}
Text: "An actively exploited remote code execution flaw affects ImageLib."
JSON: {"risk_level":"CRITICAL","concerns":["Remote code execution is actively exploited"],"evidence_quotes":["actively exploited remote code execution flaw"],"recommended_actions":["Check whether ImageLib is used and apply the vendor fix"],"breaking_changes":false}"""

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

Otherwise, use MEDIUM, HIGH, or CRITICAL for risk_level. Every concern
must have one matching evidence quote copied from the source text and one
conservative action. All three fields are arrays of plain JSON strings, never
objects. Do not invent identifiers, versions, or requirements.

{self.SEVERITY_RUBRIC}

{self.FEW_SHOT_EXAMPLES}

Return one JSON object only."""

        messages = [
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
        ]
        task_name = (
            "Risk analysis for "
            f"'{str(signal.get('title', 'Untitled'))[:80]}'"
        )
        result = self._complete(messages, signal, task_name)

        # Second look: small models often answer LOW even when the text
        # explicitly says "removed" or "deprecated". Ask once more, pointing
        # at the exact wording. This is not a heuristic override: the model's
        # second answer is final, and it may still choose LOW.
        if result["risk_level"] == "LOW":
            source_text = " ".join(
                [str(signal.get("title", "")), str(signal.get("description", ""))]
            )
            terms = self.find_explicit_risk_terms(source_text)
            if terms:
                review_messages = [
                    *messages,
                    {"role": "assistant", "content": json.dumps(result)},
                    {
                        "role": "user",
                        "content": (
                            "Check your answer again. The source text "
                            "explicitly contains: "
                            + ", ".join(f'"{term}"' for term in terms)
                            + ". If this wording describes a vulnerability, "
                            "breaking change, removal, or deprecation, return "
                            "MEDIUM, HIGH, or CRITICAL using the severity "
                            "rubric, with evidence quotes copied from the "
                            "source. If it does not describe a risk to "
                            "existing users, return LOW again. Return one "
                            "JSON object only."
                        ),
                    },
                ]
                result = self._complete(
                    review_messages,
                    signal,
                    task_name + " (second look)",
                )
        return result

    def _complete(
        self,
        messages: List[Mapping[str, str]],
        signal: Dict[str, Any],
        task_name: str,
    ) -> Dict[str, Any]:
        return complete_validated_json(
            llm_client=self._get_llm_client(),
            messages=messages,
            validator=lambda payload: self._validate_response(payload, signal),
            task_name=task_name,
            model=self.model_id,
            temperature=0.0,
            max_tokens=650,
        )

    @classmethod
    def find_explicit_risk_terms(cls, text: str) -> List[str]:
        """Return distinct explicit risk terms found in the source text."""
        found: List[str] = []
        for match in cls.EXPLICIT_RISK_TERMS.finditer(text):
            term = match.group(0)
            if term.casefold() not in {item.casefold() for item in found}:
                found.append(term)
        return found

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
