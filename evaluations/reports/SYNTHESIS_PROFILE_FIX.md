# Profile-aware synthesis — 2026-11-24

Found by running the real Streamlit app end-to-end against the self-hosted
`qwen3.5:2b` (3 sources × 2 signals, profile: Python, JavaScript / AI agents,
LLM tooling).

## Bug

`SynthesisAgent.synthesize()` and `revise()` never received the developer
profile, but the critic scores **profile relevance** against that profile. The
digest could not be personalized, so it kept failing review. In the app this
showed as a contradiction: 4 of 6 signals scored high relevance for "AI
agents", yet the summary said the evidence had *"no developer-specific
intelligence"* and the only action was *"no action required"*.

A second issue: the model chose the priority list itself and returned a single
signal scored 65 while signals scored 70+ were left out.

## Fix (`agents/synthesis_agent.py`, `workflows/reviewed_synthesis.py`)

1. The profile is passed into synthesis and both revisions, and each evidence
   item now includes the matched technologies, matched interests and
   actionability from the relevance agent.
2. Focus rules: high-relevance items are the main content, a digest is not only
   about risks, and "no action required" is only allowed when nothing is
   relevant.
3. Priority ranking is now deterministic (relevance × risk multiplier, top 5),
   as in the original design: agents for reasoning, plain code for sorting.

## Result (same app, same settings, live)

| Metric | Before | After |
|---|---:|---:|
| Profile relevance (critic) | 65 | **90** |
| Completeness (critic) | 70 | **75** |
| Groundedness (critic) | 85 | 85 |
| Recommendations | "No action required" | 3 specific, cited actions |
| Priority signals shown | 1 | 5, ordered by score |
| Final status | `needs_review` | `needs_review` |

Run cost after the fix: 18 model calls, 24,458 tokens, 280 s end-to-end.

## Honest limits

- Two single runs. The live feeds changed slightly between them (4 vs 3
  high-relevance signals), so this is not a perfectly controlled comparison.
- The report still does not pass: the critic flags one over-strong sentence
  ("aligns perfectly…"), and the summary overstates that the items "explicitly
  target Python and JavaScript".
- Revisions 1 and 2 returned the same text as the first draft. The 2B model
  does not yet use critic feedback to change the report; making revision
  effective (smaller, claim-by-claim fixes) is the next step.
