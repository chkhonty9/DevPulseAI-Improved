# DevPulseAI Improved

An improved version of the original `devpulse_ai` multi-agent signal
intelligence pipeline. The original architecture is kept; the reasoning layer
is rebuilt so that every AI decision is **real, validated, evidence-grounded,
reviewed, and measured**.

```text
GitHub · arXiv · Hacker News · Medium · Hugging Face
  -> SignalCollector        (deterministic utility, no LLM)
  -> RelevanceAgent         (AI, personalized to a developer profile)
  -> RiskAgent              (AI, every concern backed by a source quote)
  -> SynthesisAgent         (AI, every claim cites evidence IDs)
  -> CriticAgent            (deterministic ID checks + AI scoring)
  -> bounded revision       (max 2, then `needs_review`)
  -> Streamlit UI           (matches, citations, critic scores, history)
```

## What changed compared to the original

| Area | Original | Improved |
|---|---|---|
| Synthesis | `SynthesisAgent` built an LLM agent but never called it; the digest was counting and sorting. | The model writes the digest from an immutable evidence pack with stable IDs (`EV-…`). Every summary and recommendation must cite valid IDs. |
| Failure handling | Silent fallback to star/keyword heuristics, shown as if it were AI output. | One correction attempt for invalid output, then a visible, sanitized error. No heuristic result is ever substituted. |
| Output parsing | Splitting on code fences + `json.loads`. | Strict Pydantic contracts (`extra="forbid"`, `strict=True`) with cross-field rules (for example, LOW risk cannot list concerns). |
| Relevance | Generic "relevant to AI/ML developers". | Personalized by technologies, interests and experience level; matched values must exactly match the profile. |
| Risk | Free-text concerns, invented claims possible. | Each concern has a quote that must appear in the source text and one conservative action; invented CVEs, versions and migration requirements are rejected. |
| Quality loop | None. | Independent critic (groundedness, profile relevance, completeness) and up to two revisions driven by its feedback. |
| Evaluation | None. | 18 manually reviewed cases with measured agreement, invalid-output rate, groundedness, latency and tokens. |
| Providers | OpenAI only, through `agno`. | Provider-neutral `LLMClient`: GetQHSE gateway (default), OpenRouter, OpenAI, Google Gemini, Ollama Cloud. Latency and token usage are recorded; keys are redacted. |
| Tests | One mock verification script. | 66 unit tests plus `verify.py` running the real agent methods with mocked model responses. |

## Measured baseline

Model `qwen3.5:2b` on 18 reviewed cases, 2026-11-24
(details in [evaluations/reports/BASELINE.md](evaluations/reports/BASELINE.md)):

| Metric | Result |
|---|---:|
| Relevance agreement | 94.12% |
| Risk agreement | 72.22% |
| Invalid output rate | 4.65% |
| Evidence groundedness | 100.00% |
| Revision success | 0.00% (final status `needs_review`) |
| Model calls / cumulative latency | 43 / ~295 s |

These numbers are reported honestly: the small 2B model often rates explicit
removals and deprecations as LOW risk, and its digest did not pass the critic
after two revisions. The pipeline reported `needs_review` instead of
presenting a weak report as finished.

### Improvement: risk accuracy 72% → 89%

The baseline showed the 2B model rating explicit removals and deprecations as
LOW. Adding a severity rubric, four few-shot examples (different from the test
cases) and a one-time "second look" when the model says LOW about explicit
risk wording raised **risk agreement from 72.22% to 88.89%** on the same 18
cases with the same model. Details, remaining misses and limits:
[evaluations/reports/RISK_IMPROVEMENT.md](evaluations/reports/RISK_IMPROVEMENT.md).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

API keys are never hardcoded, printed, committed, or written to reports.
Either set an environment variable:

```powershell
$env:GETQHSE_API_KEY="your-key"
```

or copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
(ignored by Git) and fill in the key. In the Streamlit sidebar a user can
instead choose **Enter my own API key** and pick OpenRouter, OpenAI, Gemini
or Ollama Cloud; that key stays in the current session only.

## Run

```powershell
# Streamlit UI
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py

# CLI pipeline
.\.venv\Scripts\python.exe main.py

# Offline verification (mocked model responses, no API key needed)
$env:PYTHONUTF8='1'; .\.venv\Scripts\python.exe verify.py

# Unit tests
.\.venv\Scripts\python.exe -m unittest discover -s tests

# Evaluation (see evaluations/README.md)
.\.venv\Scripts\python.exe -m evaluations.run_evaluation --validate-only
.\.venv\Scripts\python.exe -m evaluations.run_evaluation --limit 3
.\.venv\Scripts\python.exe evaluations
un_risk_evaluation.py   # risk agent only
```

## Project layout

| Path | Purpose |
|---|---|
| `adapters/` | Source fetchers (deterministic) |
| `agents/` | Collector, relevance, risk, synthesis, critic, shared bounded-retry logic (`reasoning.py`) |
| `models.py` | Pydantic contracts shared by every stage |
| `services/` | Provider-neutral LLM clients and observability |
| `config/` | Key and model resolution |
| `workflows/reviewed_synthesis.py` | Synthesis → critic → bounded revision |
| `evaluations/` | Reviewed cases, runner, saved reports |
| `tests/` | Unit tests |

## Status

Stages 0–9 of [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) are implemented and
verified. Stage 10 (UI observability) is partly done: the UI already shows
matched profile context, evidence citations, critic scores and revision
history.

Related notes: [BASELINE_REPORT.md](BASELINE_REPORT.md),
[OPENROUTER_FOUNDATION.md](OPENROUTER_FOUNDATION.md),
[HONEST_AI_FAILURES.md](HONEST_AI_FAILURES.md).

## Next steps

1. **Smaller critic tasks**: a 2B model struggles to score three criteria at
   once (revision success is 0%). Split the critic into simple yes/no checks
   per claim ("is this claim supported by EV-x?"). When a stronger model is
   available, use it for synthesis and the critic.
2. **Parallel analysis**: run relevance and risk across signals concurrently
   (or merge them into one call per signal) to cut end-to-end latency.
3. **Per-signal failure isolation**: mark a signal `analysis_failed` and
   continue, instead of stopping the whole run.
4. **Prompt-injection defense**: wrap untrusted source text in delimiters and
   instruct the model to treat it as data only.
5. **Native structured output**: use JSON-schema response formats generated
   from the Pydantic models where the provider supports them.
6. **Larger held-out evaluation set** so prompt changes are measured on cases
   they were not designed around.
7. **Result caching** by signal and model to avoid paying twice on re-runs.
