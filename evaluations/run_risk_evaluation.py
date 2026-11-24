"""Run only the RiskAgent over the 18 reviewed cases against the live model.

Usage: python evaluations/run_risk_evaluation.py [output.json]
"""
import json
import sys
import time
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "evaluations" / "reports" / "risk-latest.json"
sys.path.insert(0, str(root))

from agents import RiskAgent  # noqa: E402
from models import Signal  # noqa: E402
from config import resolve_getqhse_settings  # noqa: E402
from services import GetQHSEClient  # noqa: E402

secrets_path = root / ".streamlit" / "secrets.toml"
secrets = tomllib.loads(secrets_path.read_text(encoding="utf-8")) if secrets_path.exists() else {}
settings = resolve_getqhse_settings(secrets=secrets)
client = GetQHSEClient(settings)


class Counting:
    def __init__(self, inner):
        self.inner, self.calls, self.tokens, self.latency = inner, 0, 0, 0

    def complete_json(self, messages, **kw):
        self.calls += 1
        r = self.inner.complete_json(messages, **kw)
        self.tokens += r.usage.total_tokens
        self.latency += r.latency_ms
        return r


counting = Counting(client)
agent = RiskAgent(model_id=settings.model, llm_client=counting)
raw = json.loads((root / "evaluations" / "cases.json").read_text(encoding="utf-8"))
cases = raw["cases"] if isinstance(raw, dict) else raw

results, agree, failed = [], 0, 0
start = time.perf_counter()
for case in cases:
    exp = case["expected"]["risk_level"]
    try:
        sig = Signal.model_validate_json(json.dumps(case["signal"])).model_dump(mode="python")
        got = agent.assess(sig)["risk_level"]
        err = None
    except Exception as exc:  # report, don't hide
        got, err = None, f"{type(exc).__name__}: {str(exc)[:160]}"
        failed += 1
    ok = got == exp
    agree += ok
    results.append({"case_id": case["id"], "expected": exp, "got": got, "agrees": ok, "error": err})
    print(f"{'OK ' if ok else 'XX '} {case['id']:<32} expected={exp:<8} got={got} {err or ''}", flush=True)

summary = {
    "model": settings.model,
    "cases": len(cases),
    "risk_agreement_pct": round(100 * agree / len(cases), 2),
    "failed_cases": failed,
    "model_calls": counting.calls,
    "total_tokens": counting.tokens,
    "cumulative_model_latency_ms": counting.latency,
    "wall_time_s": round(time.perf_counter() - start, 1),
    "case_results": results,
}
out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "case_results"}, indent=2))
