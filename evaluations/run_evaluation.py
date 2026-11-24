"""Command-line entry point for the DevPulseAI evaluation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import resolve_getqhse_settings
from evaluations.runner import EvaluationRunner, load_cases
from services import GetQHSEClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("cases.json"),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        cases = cases[: args.limit]

    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "case_count": len(cases),
                    "case_ids": [case.id for case in cases],
                },
                indent=2,
            )
        )
        return 0

    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    secret_values = (
        tomllib.loads(secrets_path.read_text(encoding="utf-8"))
        if secrets_path.exists()
        else None
    )
    settings = resolve_getqhse_settings(
        model=args.model,
        secrets=secret_values,
    )
    runner = EvaluationRunner(
        GetQHSEClient(settings),
        settings.model,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    report = runner.run(cases)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
