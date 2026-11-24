# DevPulseAI evaluation suite

`cases.json` contains 18 manually reviewed relevance and risk cases. The
runner measures relevance agreement, risk agreement, invalid model output,
evidence groundedness, latency, token usage, optional estimated cost, and
bounded-revision success.

Validate the dataset without using an API:

```powershell
.\.venv\Scripts\python.exe -m evaluations.run_evaluation --validate-only
```

Run a small live sample:

```powershell
.\.venv\Scripts\python.exe -m evaluations.run_evaluation --limit 3
```

Run all cases and save the report:

```powershell
.\.venv\Scripts\python.exe -m evaluations.run_evaluation --output evaluations\reports\getqhse-baseline.json
```

Cost remains `null` unless both token prices are supplied explicitly with
`--input-cost-per-million` and `--output-cost-per-million`.
