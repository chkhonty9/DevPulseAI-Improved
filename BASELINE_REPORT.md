# Stage 1 Baseline Report

Date: 2026-11-23

## Environment

- Python: 3.14.5
- Virtual environment: `.venv`
- Streamlit: 1.67.0
- Agno: 3.2.1

## Verification result

Command:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe verify.py
```

Result: **passed**

The baseline verification confirmed:

1. All adapters and agents import successfully.
2. `SignalCollector` remains a deterministic utility.
3. Five mock signals are normalized and deduplicated.
4. Relevance fallback scoring returns valid scores.
5. Risk fallback assessment detects the mock breaking change.
6. Synthesis returns the expected digest structure.

## UI result

The unchanged Streamlit application starts successfully at:

```text
http://localhost:8501
```

Health endpoint result: `ok`

Root page HTTP status: `200`

## Existing baseline limitations

- Verification tests heuristic fallbacks rather than real AI calls.
- `SynthesisAgent` constructs an AI agent but does not call it from
  `synthesize()`.
- The UI does not accept an OpenRouter API key.
- The UI's automated landing-page test did not complete. The landing page
  includes a remotely hosted logo, which is a candidate cause to investigate
  during the UI improvement stage.
- The section titled `Priority Signals` displays all assessed signals instead
  of the digest's actual `priority_signals` list.

No baseline application behavior was changed during this stage.
