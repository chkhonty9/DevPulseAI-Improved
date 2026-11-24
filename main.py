"""
DevPulseAI — Multi-Agent Signal Intelligence Pipeline

Demonstrates a production-style multi-agent workflow that aggregates
technical signals from multiple sources, scores them for relevance,
assesses risks, and synthesizes an actionable intelligence digest.

Architecture:
    Adapters (fetch) → SignalCollector (normalize) → RelevanceAgent (score)
    → RiskAgent (assess) → SynthesisAgent (digest) → CriticAgent (review)
    → bounded SynthesisAgent revisions (maximum two)

Design Decisions:
    - Signal collection is a utility, not an agent (deterministic work).
    - Agents are used only where reasoning is required.
    - GetQHSE provides the protected default model gateway.
    - API keys are resolved from protected runtime configuration.

Usage:
    $env:GETQHSE_API_KEY="..."
    python main.py
"""

import sys
from typing import List, Dict, Any, Optional

# Reduced default signal count for faster demo execution
DEFAULT_SIGNAL_LIMIT = 5

# Import adapters
from adapters.github import fetch_github_trending
from adapters.arxiv import fetch_arxiv_papers
from adapters.hackernews import fetch_hackernews_stories
from adapters.medium import fetch_medium_blogs
from adapters.huggingface import fetch_huggingface_models

# Import pipeline components
from agents import (
    SignalCollector,  # Utility — no LLM
    RelevanceAgent,   # Agent — provider-neutral JSON completion
    RiskAgent,        # Agent — provider-neutral JSON completion
    SynthesisAgent,   # Agent — AI synthesis constrained by evidence IDs
    CriticAgent,      # Agent — independent quality and integrity review
)
from config import MissingGetQHSEKeyError, resolve_getqhse_settings
from models import DeveloperProfile
from services import GetQHSEClient
from workflows import ReviewedSynthesisWorkflow


def collect_signals(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Collect signals from all configured sources.

    This is pure data aggregation — no LLM involved.
    """
    fetch_limit = limit if limit is not None else DEFAULT_SIGNAL_LIMIT
    print(f"\n📡 [1/4] Collecting Signals (limit: {fetch_limit} per source)...")

    signals = []

    print("  → Fetching GitHub trending repos...")
    signals.extend(fetch_github_trending(limit=fetch_limit))

    print("  → Fetching ArXiv papers...")
    signals.extend(fetch_arxiv_papers(limit=fetch_limit))

    print("  → Fetching HackerNews stories...")
    signals.extend(fetch_hackernews_stories(limit=fetch_limit))

    print("  → Fetching Medium blogs...")
    signals.extend(fetch_medium_blogs(limit=min(fetch_limit, 3)))

    print("  → Fetching HuggingFace models...")
    signals.extend(fetch_huggingface_models(limit=fetch_limit))

    print(f"  ✓ Collected {len(signals)} raw signals")
    return signals


def run_pipeline(api_key: Optional[str] = None, model: Optional[str] = None):
    """
    Execute the full signal intelligence pipeline.

    Pipeline stages:
    1. Signal Collection — Aggregate from sources (utility, no LLM)
    2. Normalization    — Deduplicate and normalize (utility, no LLM)
    3. Relevance Score  — Rate signals through the GetQHSE hosted LLM
    4. Risk Assessment  — Identify risks through the GetQHSE hosted LLM
    5. Synthesis        — Produce a cited AI intelligence digest
    6. Critic Review    — Independently score integrity and quality
    7. Bounded Revision — Apply critic feedback at most twice
    """
    print("=" * 60)
    print("🧠 DevPulseAI — Signal Intelligence Pipeline")
    print("=" * 60)

    settings = resolve_getqhse_settings(
        secrets={"GETQHSE_API_KEY": api_key} if api_key else None,
        model=model,
    )
    llm_client = GetQHSEClient(settings)
    print(f"\n🤖 GetQHSE model: {settings.model}")
    print(f"🔐 Key source: {settings.key_source}")

    # Stage 1: Collect raw signals from adapters
    raw_signals = collect_signals()

    # Stage 2: Normalize and deduplicate (utility — no LLM)
    collector = SignalCollector()
    print("\n🔄 [2/4] Normalizing Signals...")
    normalized = collector.collect(raw_signals)
    print(f"  ✓ {collector.summarize_collection(normalized)}")

    # Stage 3: Score for relevance (agent — GetQHSE hosted LLM)
    relevance = RelevanceAgent(model_id=settings.model, llm_client=llm_client)
    developer_profile = DeveloperProfile(
        technologies=["Python", "JavaScript"],
        interests=["AI agents", "LLM tooling"],
        experience_level="intermediate",
    )
    print("\n📊 [3/4] Scoring Relevance...")
    scored = relevance.score_batch(normalized, developer_profile)
    high_relevance = sum(
        1 for s in scored if s.get("relevance", {}).get("score", 0) >= 70
    )
    print(f"  ✓ {high_relevance}/{len(scored)} signals rated high-relevance")

    # Stage 4: Assess risks (agent — GetQHSE hosted LLM)
    risk = RiskAgent(model_id=settings.model, llm_client=llm_client)
    print("\n⚠️  [4/4] Assessing Risks...")
    assessed = risk.assess_batch(scored)
    critical = sum(
        1
        for s in assessed
        if s.get("risk", {}).get("risk_level") in ["HIGH", "CRITICAL"]
    )
    print(f"  ✓ {critical}/{len(assessed)} signals with elevated risk")

    # Stage 5: Synthesize an AI digest constrained to the evidence pack
    synthesis = SynthesisAgent(model_id=settings.model, llm_client=llm_client)
    print("\n📋 Generating Intelligence Digest...")
    digest = synthesis.synthesize(assessed, developer_profile)

    # Stage 6: Independently review evidence integrity and report quality
    critic = CriticAgent(model_id=settings.model, llm_client=llm_client)
    review_workflow = ReviewedSynthesisWorkflow(synthesis, critic)
    print("\n🔎 Reviewing and Revising Digest Quality...")
    pipeline_result = review_workflow.run(
        assessed,
        developer_profile,
        initial_digest=digest,
    )
    digest = pipeline_result["digest"]
    critic_result = pipeline_result["critic"]
    revisions_used = pipeline_result["revision_attempts"]
    review_status = "PASSED" if critic_result["passed"] else "NEEDS REVIEW"
    print(
        f"  ✓ Critic status: {review_status} "
        f"after {revisions_used} revision(s)"
    )

    # Output results
    print("\n" + "=" * 60)
    print("📄 INTELLIGENCE DIGEST")
    print("=" * 60)
    print(f"\n🕐 Generated: {digest['generated_at']}")
    print(f"📦 Total Signals: {digest['total_signals']}")
    print(f"\n📝 Summary: {digest['executive_summary']}")
    print(f"   Evidence: {', '.join(digest['summary_evidence_ids'])}")

    print("\n🎯 Top Priority Signals:")
    for i, signal in enumerate(digest.get("priority_signals", [])[:3], 1):
        score = signal.get("relevance", {}).get("score", "?")
        risk_level = signal.get("risk", {}).get("risk_level", "?")
        print(f"   {i}. [{signal['source']}] {signal['title'][:50]}...")
        print(f"      Relevance: {score} | Risk: {risk_level}")

    print("\n💡 Recommendations:")
    for rec in digest.get("recommendations", []):
        print(f"   • {rec['text']}")
        print(f"     Evidence: {', '.join(rec['evidence_ids'])}")

    print("\n🔎 Independent Review:")
    print(f"   Status: {review_status}")
    print(f"   Revisions used: {revisions_used}/2")
    print(
        "   Scores: "
        f"groundedness {critic_result['groundedness_score']:.0f}, "
        f"profile relevance {critic_result['profile_relevance_score']:.0f}, "
        f"completeness {critic_result['completeness_score']:.0f}"
    )
    for feedback_item in critic_result.get("feedback", []):
        print(f"   • {feedback_item}")

    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)

    return digest


if __name__ == "__main__":
    try:
        run_pipeline()
    except MissingGetQHSEKeyError as exc:
        print(f"\nConfiguration error: {exc}")
        sys.exit(2)
