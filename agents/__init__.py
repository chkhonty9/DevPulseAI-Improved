"""
DevPulseAI Agents Package

This package contains the intelligence pipeline components:

- SignalCollector: Pure utility (NOT an agent) — normalizes and deduplicates signals.
  Signal collection is deterministic and intentionally not agent-driven.

- RelevanceAgent: LLM agent — scores signals 0-100 for developer relevance.
  Uses the shared provider-neutral model client.

- RiskAgent: LLM agent — assesses security risks and breaking changes.
  Uses the shared provider-neutral model client.

- SynthesisAgent: generates an AI digest from an immutable evidence pack and
  requires citations for the summary and every recommendation.

- CriticAgent: independently scores the digest and performs deterministic
  evidence/signal ID integrity checks.

Design Principle:
    Agents are used ONLY where reasoning is required.
    Deterministic operations (collection, normalization, dedup) are utilities.
"""

from .signal_collector import SignalCollector
from .relevance_agent import RelevanceAgent
from .risk_agent import RiskAgent
from .synthesis_agent import SynthesisAgent
from .critic_agent import CriticAgent
from .reasoning import AnalysisFailedError, AgentOutputError

__all__ = [
    "SignalCollector",
    "RelevanceAgent",
    "RiskAgent",
    "SynthesisAgent",
    "CriticAgent",
    "AnalysisFailedError",
    "AgentOutputError",
]
