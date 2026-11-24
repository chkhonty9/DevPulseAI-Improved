"""Synthesis → critic → bounded revision orchestration."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from agents import AnalysisFailedError, CriticAgent, SynthesisAgent
from models import (
    ContractValidationError,
    DeveloperProfile,
    PipelineResult,
    validate_contract,
)
from services import LLMRequestError


class ReviewedSynthesisWorkflow:
    """Allow at most two evidence-grounded revisions after critic feedback."""

    MAX_REVISIONS = 2

    def __init__(
        self,
        synthesis_agent: SynthesisAgent,
        critic_agent: CriticAgent,
    ):
        self.synthesis_agent = synthesis_agent
        self.critic_agent = critic_agent

    def run(
        self,
        signals: List[Dict[str, Any]],
        profile: DeveloperProfile | Mapping[str, Any],
        *,
        initial_digest: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Review and revise until passed or the two-revision cap is reached."""
        validated_profile = (
            profile
            if isinstance(profile, DeveloperProfile)
            else validate_contract(
                DeveloperProfile,
                profile,
                context="Reviewed synthesis developer profile",
            )
        )
        digest = (
            dict(initial_digest)
            if initial_digest is not None
            else self.synthesis_agent.synthesize(signals)
        )
        history = []
        final_critic: Optional[Dict[str, Any]] = None
        revision_attempts = 0
        revision_failure: Optional[str] = None

        for revision_number in range(self.MAX_REVISIONS + 1):
            final_critic = self.critic_agent.review(digest, validated_profile)
            history.append(
                {
                    "revision_number": revision_number,
                    "digest_summary": str(digest.get("executive_summary", "")),
                    "critic": final_critic,
                }
            )
            if final_critic["passed"]:
                break
            if revision_number == self.MAX_REVISIONS:
                break
            revision_attempts += 1
            try:
                digest = self.synthesis_agent.revise(
                    signals,
                    digest,
                    final_critic,
                    revision_number=revision_number + 1,
                )
            except (
                AnalysisFailedError,
                ContractValidationError,
                LLMRequestError,
            ) as exc:
                revision_failure = (
                    f"Revision {revision_attempts} could not produce a valid "
                    f"digest: {exc}"
                )
                final_critic = dict(final_critic)
                final_critic["feedback"] = [
                    *final_critic.get("feedback", []),
                    revision_failure,
                ]
                history[-1]["critic"] = final_critic
                break

        if final_critic is None:
            raise RuntimeError("The reviewed synthesis workflow produced no review.")

        payload = {
            "status": "completed" if final_critic["passed"] else "needs_review",
            "signals": signals,
            "digest": digest,
            "critic": final_critic,
            "evidence": digest.get("evidence_pack", []),
            "revision_history": history,
            "revision_attempts": revision_attempts,
            "revision_failure": revision_failure,
        }
        result = PipelineResult.model_validate_json(
            json.dumps(
                payload,
                default=lambda value: value.isoformat()
                if hasattr(value, "isoformat")
                else str(value),
            )
        )
        return result.model_dump(mode="json")
