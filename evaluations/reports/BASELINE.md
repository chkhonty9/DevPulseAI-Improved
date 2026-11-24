# GetQHSE baseline — 2026-11-24

Model: `qwen3.5:2b`  
Dataset: 18 manually reviewed cases  
Raw report: `getqhse-baseline-18.json`

## Measured results

| Metric | Result |
|---|---:|
| Relevance agreement | 94.12% |
| Risk agreement | 72.22% |
| Invalid output rate | 4.65% |
| Evidence groundedness | 100.00% |
| Provider request errors | 0 |
| Model calls | 43 |
| Cumulative model latency | 295,461 ms |
| Prompt tokens | 13,102 |
| Completion tokens | 4,140 |
| Total tokens | 17,242 |
| Estimated cost | Not configured |
| Revision attempts | 2 |
| Revision success | 0.00% |
| Final pipeline status | `needs_review` |

## Observed weaknesses

- Five risk cases disagreed with the reviewed expectation. The small model
  frequently returned `LOW` for explicit removals or deprecations.
- One relevance case scored 65 although its reviewed acceptable range began at
  70.
- One relevance case failed validation because the model returned matched
  profile values that were not exact copies of the supplied profile.
- The final digest remained below the critic threshold after two revisions.

## Interpretation limits

- Relevance agreement uses the 17 cases that produced a valid relevance
  result; the invalid case is reflected separately in invalid-output rate.
- Risk agreement uses all 18 cases.
- Revision success is based on one end-to-end digest trial in this run, not 18
  independent revision trials.
- Cost is intentionally absent because no verified input/output token prices
  were supplied for the hosted gateway.
- Model outputs can vary between runs; this file is a dated baseline, while the
  JSON report preserves the exact per-case observations.
