"""Merge disjoint benchmark shards and recompute paired ablation metrics."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from coscope.evaluation.sharing_ablation import reaggregate_sharing_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="BENCHMARK=REPORT",
        help="replace one benchmark with the matching shard from REPORT",
    )
    parser.add_argument(
        "--output-cap",
        action="append",
        default=[],
        metavar="BENCHMARK=TOKENS",
        help="record an explicit benchmark-specific output cap",
    )
    parser.add_argument(
        "--append",
        action="append",
        default=[],
        metavar="REPORT",
        help="append a disjoint paired shard for every benchmark in REPORT",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")

    base_path = Path(args.base)
    report = _read_report(base_path)
    sources: dict[str, Any] = {
        "base": str(base_path),
        "replacements": {},
        "appended": [],
    }
    for replacement in args.replace:
        benchmark, separator, raw_path = replacement.partition("=")
        if not separator or not benchmark or not raw_path:
            parser.error("--replace must use BENCHMARK=REPORT")
        replacement_path = Path(raw_path)
        shard = _read_report(replacement_path)
        if benchmark not in shard["benchmarks"]:
            parser.error(f"{benchmark!r} is absent from {replacement_path}")
        report["benchmarks"][benchmark] = deepcopy(
            shard["benchmarks"][benchmark]
        )
        sources["replacements"][benchmark] = str(replacement_path)
        report["models"].setdefault(
            "max_output_tokens_by_benchmark", {}
        )[benchmark] = (
            shard.get("models", {})
            .get("max_output_tokens_by_benchmark", {})
            .get(
                benchmark,
                shard.get("run_config", {}).get(
                    "math_max_output_tokens"
                ),
            )
        )
    for raw_path in args.append:
        shard_path = Path(raw_path)
        shard = _read_report(shard_path)
        _append_disjoint_shard(report, shard, source=shard_path)
        sources["appended"].append(str(shard_path))
    for raw_cap in args.output_cap:
        benchmark, separator, raw_tokens = raw_cap.partition("=")
        if not separator or not benchmark:
            parser.error("--output-cap must use BENCHMARK=TOKENS")
        try:
            tokens = int(raw_tokens)
        except ValueError:
            parser.error("--output-cap TOKENS must be an integer")
        if tokens <= 0:
            parser.error("--output-cap TOKENS must be positive")
        report["models"].setdefault(
            "max_output_tokens_by_benchmark", {}
        )[benchmark] = tokens

    report.setdefault("design", {})["composite_sources"] = sources
    report["run_config"] = {
        **report.get("run_config", {}),
        "composite": True,
        "bootstrap_samples": args.bootstrap_samples,
    }
    merged = reaggregate_sharing_report(
        report,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "overall": merged["overall"],
                "comparisons": merged["comparisons"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _read_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("benchmarks"), dict
    ):
        raise ValueError(f"{path} is not a sharing ablation report")
    return payload


def _append_disjoint_shard(
    report: dict[str, Any],
    shard: dict[str, Any],
    *,
    source: Path,
) -> None:
    for benchmark, shard_benchmark in shard["benchmarks"].items():
        if benchmark not in report["benchmarks"]:
            report["benchmarks"][benchmark] = deepcopy(shard_benchmark)
            continue
        target_benchmark = report["benchmarks"][benchmark]
        for arm in ("coscope", "full_sharing", "no_sharing"):
            incoming = shard_benchmark[arm]["tasks"]
            existing = {
                task["example_id"]
                for task in target_benchmark[arm]["tasks"]
            }
            overlap = sorted(
                existing
                & {task["example_id"] for task in incoming}
            )
            if overlap:
                joined = ", ".join(overlap[:3])
                raise ValueError(
                    f"{source} repeats {benchmark}/{arm}: {joined}"
                )
            target_benchmark[arm]["tasks"].extend(deepcopy(incoming))


if __name__ == "__main__":
    raise SystemExit(main())
