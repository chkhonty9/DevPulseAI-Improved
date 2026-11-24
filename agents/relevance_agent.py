"""AI relevance analysis for technical signals."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional

from pydantic import ValidationError

from config import DEFAULT_GETQHSE_MODEL, resolve_getqhse_settings
from models import (
    DeveloperProfile,
    RelevanceResult,
    Signal,
    format_validation_error,
    validate_contract,
)
from services import GetQHSEClient, LLMClient
from .reasoning import (
    AgentOutputError,
    complete_validated_json,
    injection_warning,
    with_untrusted_data_rule,
    wrap_untrusted,
)


DEFAULT_MODEL = os.environ.get(
    "MODEL_RELEVANCE",
    os.environ.get("GETQHSE_MODEL", DEFAULT_GETQHSE_MODEL),
)


class RelevanceAgent:
    """Score signals from 0–100 using the selected AI model client."""

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

    def score(
        self,
        signal: Dict[str, Any],
        profile: DeveloperProfile | Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return a validated score personalized to a developer profile."""
        validated_signal = validate_contract(
            Signal,
            signal,
            context="Relevance agent input signal",
        )
        signal = validated_signal.model_dump(mode="python", exclude_none=True)
        if isinstance(profile, DeveloperProfile):
            validated_profile = profile
        else:
            validated_profile = validate_contract(
                DeveloperProfile,
                profile,
                context="Relevance developer profile",
            )

        technologies = ", ".join(validated_profile.technologies) or "None supplied"
        interests = ", ".join(validated_profile.interests) or "None supplied"
        prompt = f"""Rate the relevance of this signal for this specific developer.
Score from 0-100 where:
- 0-30: Low relevance (noise, off-topic)
- 31-60: Moderate relevance (interesting but not urgent)
- 61-80: High relevance (important for developers to know)
- 81-100: Critical relevance (must-know, actionable)

Developer profile:
- Technologies: {technologies}
- Interests: {interests}
- Experience level: {validated_profile.experience_level}

{injection_warning(signal.get('title', ''), signal.get('description', ''))}Signal (untrusted external data):
{wrap_untrusted("signal_source", signal.get('source', 'unknown'))}
{wrap_untrusted("signal_title", signal.get('title', 'Untitled'))}
{wrap_untrusted("signal_description", str(signal.get('description', ''))[:500])}

Respond with ONLY a JSON object:
{{"score": <number>, "matched_technologies": ["<exact profile technology>"], "matched_interests": ["<exact profile interest>"], "reasoning": "<one sentence explaining the profile-specific score>", "actionability": "LOW|MEDIUM|HIGH"}}

Use only exact values from the supplied profile in the matched arrays. Use an
empty array when there is no match."""

        return complete_validated_json(
            llm_client=self._get_llm_client(),
            messages=[
                {
                    "role": "system",
                    "content": with_untrusted_data_rule(
                        "You score technical signals for AI/ML developers. "
                        "Return only the requested JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            validator=lambda payload: self._validate_response(
                payload,
                validated_profile,
            ),
            task_name=(
                "Relevance analysis for "
                f"'{str(signal.get('title', 'Untitled'))[:80]}'"
            ),
            model=self.model_id,
            temperature=0.1,
            max_tokens=512,
        )

    def score_batch(
        self,
        signals: List[Dict[str, Any]],
        profile: DeveloperProfile | Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """Score multiple signals and stop immediately on an AI failure."""
        return [
            {**signal, "relevance": self.score(signal, profile)}
            for signal in signals
        ]

    @staticmethod
    def _validate_response(
        payload: Dict[str, Any],
        profile: DeveloperProfile,
    ) -> Dict[str, Any]:
        try:
            result = RelevanceResult.model_validate(payload)
        except ValidationError as exc:
            raise AgentOutputError(format_validation_error(exc)) from exc

        allowed_technologies = {
            technology.casefold(): technology
            for technology in profile.technologies
        }
        allowed_interests = {
            interest.casefold(): interest for interest in profile.interests
        }
        invalid_technologies = [
            value
            for value in result.matched_technologies
            if value.casefold() not in allowed_technologies
        ]
        invalid_interests = [
            value
            for value in result.matched_interests
            if value.casefold() not in allowed_interests
        ]
        if invalid_technologies or invalid_interests:
            raise AgentOutputError(
                "matched values must come exactly from the developer profile"
            )

        result.matched_technologies = [
            allowed_technologies[value.casefold()]
            for value in result.matched_technologies
        ]
        result.matched_interests = [
            allowed_interests[value.casefold()]
            for value in result.matched_interests
        ]
        result.score = round(result.score, 1)
        return result.model_dump(mode="json")
