"""Run a live GSM8K MAS experiment across retrieval thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coscope.config import CoScopeSettings
from coscope.evaluation.benchmarks import load_gsm8k
from coscope.evaluation.mas_experiment import (
    ThresholdPoint,
    run_gsm8k_threshold_sweep,
)


def _thresholds(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 or item > 1 for item in values):
        raise argparse.ArgumentTypeError(
            "thresholds must be a comma-separated list between 0 and 1"
        )
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="raw/gsm8k/test.parquet")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--thresholds", type=_thresholds, default=[0.75, 0.82, 0.90])
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument(
        "--output",
        default="new_results/mas_gsm8k_threshold_sweep.json",
    )
    args = parser.parse_args()
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive when set")

    examples = load_gsm8k(args.dataset, limit=args.limit, seed=args.seed)
    settings = CoScopeSettings.from_env(".env")
    points = [ThresholdPoint(value, value) for value in args.thresholds]
    report = run_gsm8k_threshold_sweep(
        examples,
        settings,
        points,
        max_output_tokens=args.max_output_tokens,
    )
    report["run_config"] = {
        "seed": args.seed,
        "limit": args.limit,
        "max_output_tokens": args.max_output_tokens,
        "dataset": args.dataset,
        "purpose": "functional_validation",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "output": str(output),
            "threshold_results": [
                {
                    "thresholds": point["thresholds"],
                    "aggregate": point["aggregate"],
                }
                for point in report["threshold_points"]
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
