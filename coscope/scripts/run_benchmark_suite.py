"""Run live MAS validation on all locally available benchmark families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coscope.config import CoScopeSettings
from coscope.evaluation.benchmark_registry import (
    BENCHMARK_LOADERS,
    load_selected_benchmarks,
    parse_benchmark_limits,
    resolve_benchmark_limits,
)
from coscope.evaluation.benchmark_suite import run_benchmark_suite
from coscope.evaluation.code_benchmark import (
    DEFAULT_EVALPLUS_IMAGE,
    EvalPlusDockerEvaluator,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--benchmark-limits",
        default="",
        help="per-benchmark overrides, for example aime2024=30,mbpp_plus=378",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--musique-max-output-tokens", type=int)
    parser.add_argument("--aime-max-output-tokens", type=int)
    parser.add_argument("--mbpp-max-output-tokens", type=int)
    parser.add_argument(
        "--benchmarks",
        default="hotpotqa,2wikimultihopqa,musique,math",
        help="comma-separated benchmark names",
    )
    parser.add_argument(
        "--evalplus-image",
        default=DEFAULT_EVALPLUS_IMAGE,
    )
    parser.add_argument(
        "--evalplus-artifact-dir",
        default="new_results/evalplus",
    )
    parser.add_argument("--evalplus-parallel", type=int, default=2)
    parser.add_argument(
        "--output",
        default="new_results/mas_benchmark_suite.json",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if any(
        value is not None and value <= 0
        for value in (
            args.max_output_tokens,
            args.musique_max_output_tokens,
            args.aime_max_output_tokens,
            args.mbpp_max_output_tokens,
            args.evalplus_parallel,
        )
    ):
        parser.error("token limits and EvalPlus parallelism must be positive")
    requested = [
        name.strip() for name in args.benchmarks.split(",") if name.strip()
    ]
    unknown = sorted(set(requested) - BENCHMARK_LOADERS.keys())
    if unknown:
        parser.error(
            f"unknown benchmarks: {', '.join(unknown)}; "
            f"choose from {', '.join(BENCHMARK_LOADERS)}"
        )
    if not requested:
        parser.error("--benchmarks must contain at least one benchmark")
    try:
        limit_overrides = parse_benchmark_limits(
            args.benchmark_limits,
            allowed=set(requested),
        )
        resolved_limits = resolve_benchmark_limits(
            requested,
            default_limit=args.limit,
            overrides=limit_overrides,
        )
    except ValueError as error:
        parser.error(str(error))

    examples = load_selected_benchmarks(
        requested,
        limit=args.limit,
        seed=args.seed,
        limits_by_benchmark=limit_overrides,
    )
    code_evaluator = (
        EvalPlusDockerEvaluator(
            BENCHMARK_LOADERS["mbpp_plus"][1],
            artifact_root=args.evalplus_artifact_dir,
            image=args.evalplus_image,
            parallel=args.evalplus_parallel,
        )
        if "mbpp_plus" in requested
        else None
    )
    report = run_benchmark_suite(
        examples,
        CoScopeSettings.from_env(".env"),
        threshold=args.threshold,
        max_output_tokens=args.max_output_tokens,
        max_output_tokens_by_benchmark={
            benchmark: cap
            for benchmark, cap in {
                "musique": args.musique_max_output_tokens,
                "aime2024": args.aime_max_output_tokens,
                "aime2025": args.aime_max_output_tokens,
                "mbpp_plus": args.mbpp_max_output_tokens,
            }.items()
            if cap is not None
        },
        code_evaluator=code_evaluator,
    )
    report["run_config"] = {
        "limit_per_benchmark": args.limit,
        "limits_by_benchmark": resolved_limits,
        "seed": args.seed,
        "benchmarks": requested,
        "musique_max_output_tokens": args.musique_max_output_tokens,
        "aime_max_output_tokens": args.aime_max_output_tokens,
        "mbpp_max_output_tokens": args.mbpp_max_output_tokens,
        "evalplus_image": (
            args.evalplus_image if "mbpp_plus" in requested else None
        ),
        "purpose": "functional_validation",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "benchmarks": {
                    name: result["aggregate"]
                    for name, result in report["benchmarks"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
