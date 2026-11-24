"""
DevPulseAI Verification Script

Verifies the complete pipeline using MOCK DATA ONLY.
No network calls, no API keys, no external dependencies required.

Usage:
    python verify.py

Expected output:
    [OK] DevPulseAI reference pipeline executed successfully

Runs in <1s on any machine.
"""

import json
import re
import sys
import time
from typing import List, Dict, Any

from services import LLMResult
from models import DeveloperProfile

# ────────────────────────────────────────────────
# Mock signal data — representative of real adapter output
# ────────────────────────────────────────────────
MOCK_SIGNALS = [
    {
        "id": "mock-gh-001",
        "source": "github",
        "title": "awesome-llm-apps",
        "description": "A curated collection of awesome LLM apps built with RAG and AI agents.",
        "url": "https://github.com/Shubhamsaboo/awesome-llm-apps",
        "metadata": {"stars": 5000, "language": "Python", "topics": ["llm", "ai"]},
    },
    {
        "id": "mock-arxiv-001",
        "source": "arxiv",
        "title": "Attention Is All You Need: Revisited",
        "description": "A comprehensive analysis of transformer architectures.",
        "url": "https://arxiv.org/abs/2401.00001",
        "metadata": {"pdf": "https://arxiv.org/pdf/2401.00001", "published": "2024-01-15"},
    },
    {
        "id": "mock-hn-001",
        "source": "hackernews",
        "title": "GPT-5 Breaking Changes in API",
        "description": "OpenAI announces breaking changes to the Chat Completions API.",
        "url": "https://news.ycombinator.com/item?id=12345",
        "metadata": {"points": 500, "comments": 200, "author": "techwriter"},
    },
    {
        "id": "mock-medium-001",
        "source": "medium",
        "title": "Building Production RAG Systems",
        "description": "A deep dive into building scalable retrieval-augmented generation.",
        "url": "https://medium.com/@techblog/building-rag",
        "metadata": {"author": "TechBlog", "published": "2024-01-20"},
    },
    {
        "id": "mock-hf-001",
        "source": "huggingface",
        "title": "HF Model: meta-llama/Llama-3-8B",
        "description": "Pipeline: text-generation | Downloads: 1,000,000 | Likes: 5,000",
        "url": "https://huggingface.co/meta-llama/Llama-3-8B",
        "metadata": {"downloads": 1000000, "likes": 5000, "pipeline_tag": "text-generation"},
    },
]


class MockLLMClient:
    """Deterministic test double that exercises the real agent code paths."""

    def complete_json(self, messages, **kwargs):
        system_prompt = messages[0]["content"]
        user_prompt = messages[-1]["content"]

        if "score technical signals" in system_prompt:
            payload = {
                "score": 78,
                "matched_technologies": ["Python"],
                "matched_interests": ["AI agents"],
                "reasoning": "Mock AI response for pipeline verification.",
                "actionability": "HIGH",
            }
        elif "independent evidence and relevance critic" in system_prompt:
            payload = {
                "groundedness_score": 94.0,
                "profile_relevance_score": 88.0,
                "completeness_score": 86.0,
                "unsupported_claims": [],
                "feedback": ["Keep evidence citations visible."],
            }
        elif "synthesize developer intelligence" in system_prompt:
            evidence_ids = list(
                dict.fromkeys(re.findall(r"EV-[A-F0-9]{12}", user_prompt))
            )
            payload = {
                "executive_summary": (
                    "The assessed signals contain relevant developer updates "
                    "and one compatibility concern."
                ),
                "summary_evidence_ids": evidence_ids[:2],
                "priority_evidence_ids": evidence_ids[:3],
                "recommendations": [
                    {
                        "text": "Review the highest-priority compatibility signal.",
                        "evidence_ids": [evidence_ids[0]],
                    }
                ],
            }
        else:
            breaking = (
                "<signal_title>\nGPT-5 Breaking Changes in API\n</signal_title>"
                in user_prompt
            )
            payload = {
                "risk_level": "MEDIUM" if breaking else "LOW",
                "concerns": ["Potential compatibility impact"] if breaking else [],
                "evidence_quotes": ["breaking changes"] if breaking else [],
                "recommended_actions": (
                    ["Review compatibility impact before updating."]
                    if breaking
                    else []
                ),
                "breaking_changes": breaking,
            }

        return LLMResult(
            content=json.dumps(payload),
            model=kwargs.get("model") or "mock/model",
            latency_ms=1,
        )


# ────────────────────────────────────────────────
# Verification steps
# ────────────────────────────────────────────────

def verify_imports():
    """Verify all modules can be imported without errors."""
    print("[1/6] Verifying imports...")

    from agents import (
        CriticAgent,
        RelevanceAgent,
        RiskAgent,
        SignalCollector,
        SynthesisAgent,
    )
    from adapters.github import fetch_github_trending
    from adapters.arxiv import fetch_arxiv_papers
    from adapters.hackernews import fetch_hackernews_stories
    from adapters.medium import fetch_medium_blogs
    from adapters.huggingface import fetch_huggingface_models

    # Verify SignalCollector is NOT an agent (no .agent attribute)
    collector = SignalCollector()
    assert not hasattr(collector, "agent"), (
        "SignalCollector should NOT have an .agent attribute — "
        "it's a utility, not an agent"
    )

    print("  ✓ All modules imported successfully")
    print("  ✓ SignalCollector confirmed as utility (no LLM)")
    return True


def verify_signal_collector():
    """Verify SignalCollector normalizes and deduplicates correctly."""
    print("[2/6] Verifying Signal Collector (utility)...")

    from agents import SignalCollector

    collector = SignalCollector()

    # Test normalization
    normalized = collector.collect(MOCK_SIGNALS)
    assert len(normalized) == len(MOCK_SIGNALS), "Signal count mismatch"
    assert all("collected_at" in s for s in normalized), "Missing timestamp"

    # Test deduplication — adding duplicates should not increase count
    duped = MOCK_SIGNALS + [MOCK_SIGNALS[0]]  # One duplicate
    deduped = collector.collect(duped)
    assert len(deduped) == len(MOCK_SIGNALS), "Deduplication failed"

    summary = collector.summarize_collection(normalized)
    print(f"  ✓ {summary}")
    print("  ✓ Deduplication works correctly")
    return normalized


def verify_relevance_agent(signals: List[Dict]):
    """Verify the actual RelevanceAgent path with mocked AI responses."""
    print("[3/6] Verifying Relevance Agent (mock AI)...")

    from agents import RelevanceAgent

    agent = RelevanceAgent(model_id="mock/model", llm_client=MockLLMClient())
    profile = DeveloperProfile(
        technologies=["Python"],
        interests=["AI agents"],
        experience_level="intermediate",
    )
    scored = agent.score_batch(signals, profile)

    assert all("relevance" in s for s in scored), "Missing relevance scores"
    assert all(
        0 <= s["relevance"]["score"] <= 100 for s in scored
    ), "Scores out of range"

    print(f"  ✓ Scored {len(scored)} signals through the agent's AI path")
    return scored


def verify_risk_agent(signals: List[Dict]):
    """Verify the actual RiskAgent path with mocked AI responses."""
    print("[4/6] Verifying Risk Agent (mock AI)...")

    from agents import RiskAgent

    agent = RiskAgent(model_id="mock/model", llm_client=MockLLMClient())
    assessed = agent.assess_batch(signals)

    assert all("risk" in s for s in assessed), "Missing risk assessments"

    # The mock model flags the supplied "Breaking Changes" signal.
    breaking = [s for s in assessed if s.get("risk", {}).get("breaking_changes")]
    assert len(breaking) >= 1, "Breaking change detection failed"

    print(f"  ✓ Assessed {len(assessed)} signals ({len(breaking)} with breaking changes)")
    return assessed


def verify_synthesis_agent(signals: List[Dict]):
    """Verify SynthesisAgent produces valid digest structure."""
    print("[5/6] Verifying Synthesis Agent...")

    from agents import SynthesisAgent

    agent = SynthesisAgent(model_id="mock/model", llm_client=MockLLMClient())
    digest = agent.synthesize(signals)

    assert "generated_at" in digest, "Missing timestamp"
    assert "executive_summary" in digest, "Missing summary"
    assert "recommendations" in digest, "Missing recommendations"
    assert "priority_signals" in digest, "Missing priority signals"
    assert "evidence_pack" in digest, "Missing evidence pack"
    assert digest["total_signals"] == len(signals), "Signal count mismatch"
    known_ids = {item["evidence_id"] for item in digest["evidence_pack"]}
    assert set(digest["summary_evidence_ids"]) <= known_ids
    assert all(
        recommendation["evidence_ids"]
        and set(recommendation["evidence_ids"]) <= known_ids
        for recommendation in digest["recommendations"]
    ), "Recommendation citations are invalid"

    print(f"  ✓ Generated digest with {len(digest['recommendations'])} recommendations")
    return digest


def verify_critic_agent(digest: Dict[str, Any]):
    """Verify the independent AI critic and deterministic checks."""
    print("[6/6] Verifying Independent Critic...")

    from agents import CriticAgent

    profile = DeveloperProfile(
        technologies=["Python"],
        interests=["AI agents"],
        experience_level="intermediate",
    )
    critic = CriticAgent(model_id="mock/model", llm_client=MockLLMClient())
    result = critic.review(digest, profile)

    assert result["passed"], "Valid digest should pass the mock critic"
    assert result["groundedness_score"] >= 70
    assert not result["unsupported_claims"]

    print(
        "  ✓ Critic passed with groundedness "
        f"{result['groundedness_score']:.0f}/100"
    )
    return result


# ────────────────────────────────────────────────
# Main verification runner
# ────────────────────────────────────────────────

def run_verification():
    """Run the complete verification suite with timing."""
    print("=" * 60)
    print("🔍 DevPulseAI Verification Suite")
    print("=" * 60)
    print("\nUsing MOCK DATA — No network calls or API keys required.\n")

    start = time.time()

    try:
        if not verify_imports():
            raise AssertionError("Import verification failed")

        normalized = verify_signal_collector()
        scored = verify_relevance_agent(normalized)
        assessed = verify_risk_agent(scored)
        digest = verify_synthesis_agent(assessed)
        critic = verify_critic_agent(digest)

        elapsed = time.time() - start

        # Final summary
        print("\n" + "=" * 60)
        print("📊 Verification Summary")
        print("=" * 60)
        print(f"  • Signals processed:  {digest['total_signals']}")
        print(f"  • Summary:            {digest['executive_summary']}")
        print(f"  • Recommendations:    {len(digest['recommendations'])}")
        print(
            "  • Critic status:      "
            + ("passed" if critic["passed"] else "needs review")
        )
        print(f"  • Time elapsed:       {elapsed:.3f}s")

        print("\n" + "=" * 60)
        print("[OK] DevPulseAI reference pipeline executed successfully")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[FAIL] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
