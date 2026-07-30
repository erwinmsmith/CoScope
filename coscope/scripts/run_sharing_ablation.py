"""Run paired CoScope, full-sharing, and no-sharing live controls."""

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
from coscope.evaluation.code_benchmark import (
    DEFAULT_EVALPLUS_IMAGE,
    EvalPlusDockerEvaluator,
)
from coscope.evaluation.sharing_ablation import run_sharing_ablation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--benchmark-limits",
        default="",
        help="per-benchmark overrides, for example aime2024=30,mbpp_plus=378",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--musique-max-output-tokens", type=int)
    parser.add_argument(
        "--math-max-output-tokens",
        type=int,
        help="optional MATH-specific output cap",
    )
    parser.add_argument("--aime-max-output-tokens", type=int)
    parser.add_argument("--mbpp-max-output-tokens", type=int)
    parser.add_argument(
        "--evalplus-image",
        default=DEFAULT_EVALPLUS_IMAGE,
    )
    parser.add_argument(
        "--evalplus-artifact-dir",
        default="new_results/evalplus",
    )
    parser.add_argument("--evalplus-parallel", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument(
        "--benchmarks",
        default="gsm8k,hotpotqa,2wikimultihopqa,musique,math",
        help="comma-separated benchmark names",
    )
    parser.add_argument(
        "--output",
        default="new_results/sharing_ablation.json",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if any(
        value is not None and value <= 0
        for value in (
            args.max_output_tokens,
            args.musique_max_output_tokens,
            args.math_max_output_tokens,
            args.aime_max_output_tokens,
            args.mbpp_max_output_tokens,
            args.evalplus_parallel,
        )
    ):
        parser.error("token limits and EvalPlus parallelism must be positive")

    requested = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
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
    report = run_sharing_ablation(
        examples,
        CoScopeSettings.from_env(".env"),
        threshold=args.threshold,
        max_output_tokens=args.max_output_tokens,
        max_output_tokens_by_benchmark={
            benchmark: cap
            for benchmark, cap in {
                "musique": args.musique_max_output_tokens,
                "math": args.math_max_output_tokens,
                "aime2024": args.aime_max_output_tokens,
                "aime2025": args.aime_max_output_tokens,
                "mbpp_plus": args.mbpp_max_output_tokens,
            }.items()
            if cap is not None
        },
        bootstrap_samples=args.bootstrap_samples,
        code_evaluator=code_evaluator,
        progress=lambda event: print(
            json.dumps({"progress": event}, ensure_ascii=False),
            flush=True,
        ),
    )
    report["run_config"] = {
        "limit_per_benchmark": args.limit,
        "limits_by_benchmark": resolved_limits,
        "seed": args.seed,
        "benchmarks": requested,
        "max_output_tokens": args.max_output_tokens,
        "musique_max_output_tokens": args.musique_max_output_tokens,
        "math_max_output_tokens": args.math_max_output_tokens,
        "aime_max_output_tokens": args.aime_max_output_tokens,
        "mbpp_max_output_tokens": args.mbpp_max_output_tokens,
        "evalplus_image": (
            args.evalplus_image if "mbpp_plus" in requested else None
        ),
        "purpose": "paired_sharing_ablation",
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
                "overall": report["overall"],
                "comparisons": report["comparisons"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
