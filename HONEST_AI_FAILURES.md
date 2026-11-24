# Honest AI Failure Handling

## Production behavior

Relevance and risk analysis now have only two valid outcomes:

1. A schema-shaped response produced by the configured OpenRouter model.
2. A visible error that stops the pipeline.

The application no longer presents popularity or keyword heuristics as if
they were AI analysis.

## Bounded retries

- The OpenAI-compatible SDK is configured for one retry of temporary request
  failures.
- If the provider returns malformed JSON or invalid fields, the agent asks for
  one corrected response.
- A second invalid response raises `AnalysisFailedError` and stops the run.
- Authentication and credit failures are not repeatedly retried.

## Error presentation

The UI distinguishes common OpenRouter failures:

- `401`: invalid or disabled API key
- `402`: insufficient credits or a key spending limit
- `429`: rate limit; common with free models
- `503` / `529`: temporary provider or model unavailability

Displayed messages are sanitized, and API keys are redacted.

## Verification

- Unit tests cover successful analysis, one correction, correction failure,
  provider failure propagation, API-key redaction, and rate-limit messages.
- `verify.py` now uses deterministic mocked AI responses through the real
  agent methods. It no longer calls private heuristic fallback methods.
- A live `openrouter/free` check produced valid relevance and risk objects for
  one sample breaking-change signal.
