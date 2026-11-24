# Risk accuracy improvement — 2026-11-24

Model: `qwen3.5:2b` (self-hosted), same 18 reviewed cases as the baseline.  
Raw report: `risk-fewshot-guard-18.json`  
Reproduce: `.\.venv\Scripts\python.exe evaluations\run_risk_evaluation.py`

## Problem found in the baseline

In 5 of the 6 risk disagreements, the model returned `LOW` for text that
explicitly said a feature was *removed* or *deprecated*. A 2B model follows
examples much better than abstract rules.

## Change (`agents/risk_agent.py`)

1. **Severity rubric** in the prompt: what makes CRITICAL, HIGH, MEDIUM, LOW.
2. **Four few-shot examples** (one per level). They are deliberately different
   from the evaluation cases, so the prompt is not tuned to the test set.
3. **Second look**: if the model answers `LOW` but the source contains explicit
   risk wording (removed, deprecated, breaking, incompatible, vulnerability,
   CVE, …), the model is asked once more and shown the exact words. The
   model's second answer is final and may still be `LOW`. This is a check, not
   a heuristic override, so the "no silent fallback" rule still holds.

## Result

| Metric | Baseline | After |
|---|---:|---:|
| Risk agreement | 72.22% (13/18) | **88.89% (16/18)** |
| Explicit removal/deprecation rated LOW | 5 | **0** |
| Risk cases failed visibly | 0 | 1 |
| Risk model calls (18 cases) | not measured separately | 19 |

## Remaining disagreements (reported, not hidden)

- `dotnet-breaking-change`: model said MEDIUM, reviewer said HIGH. This is a
  borderline severity call.
- `kubernetes-vulnerability`: the model wrote version `1.34.3` while the source
  says `1.34.2`. The existing anti-hallucination check rejected it twice, so
  the case **failed visibly** instead of showing an invented version. This was
  correct behaviour, but it costs one case.

## Honest limits

- One run; model output varies between runs.
- The few-shot examples make each risk prompt longer, so each call costs more
  tokens and latency (~8 s per call on this server).
- In this run the gain came from the rubric and examples; the second-look
  check did not need to fire. It stays as a safety net.
- 18 cases is a small dataset; a larger, held-out set is the next step.
