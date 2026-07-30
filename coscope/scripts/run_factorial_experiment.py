"""Run the uniform reasoning-mode x sharing-policy benchmark matrix."""

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
from coscope.evaluation.factorial_experiment import (
    FACTORIAL_MODE_SPECS,
    parse_batch_modes,
    parse_reasoning_modes,
    parse_sharing_arms,
    run_factorial_experiment,
)
from coscope.evaluation.mas_experiment import AGENTS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        default=",".join(BENCHMARK_LOADERS),
        help="comma-separated benchmark names",
    )
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--benchmark-limits",
        default="",
        help="per-benchmark overrides, for example aime2024=30,mbpp_plus=378",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="use the complete local split for every selected benchmark",
    )
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="required with --full because the default matrix is very costly",
    )
    parser.add_argument("--reasoning-modes", default="cot,tot")
    parser.add_argument(
        "--sharing-policies",
        default="coscope,full_sharing,no_sharing",
    )
    parser.add_argument("--batch-modes", default="batched,independent")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--musique-max-output-tokens", type=int)
    parser.add_argument("--math-max-output-tokens", type=int)
    parser.add_argument("--aime-max-output-tokens", type=int)
    parser.add_argument("--mbpp-max-output-tokens", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--evalplus-image", default=DEFAULT_EVALPLUS_IMAGE)
    parser.add_argument(
        "--evalplus-artifact-dir",
        default="new_results/evalplus_factorial",
    )
    parser.add_argument("--evalplus-parallel", type=int, default=2)
    parser.add_argument(
        "--output",
        default="new_results/factorial_experiment.json",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.full and not args.confirm_full_run:
        parser.error("--full requires --confirm-full-run; run the preflight first")
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
        reasoning_modes = parse_reasoning_modes(args.reasoning_modes)
        sharing_arms = parse_sharing_arms(args.sharing_policies)
        batch_modes = parse_batch_modes(args.batch_modes)
        limit_overrides = parse_benchmark_limits(
            args.benchmark_limits,
            allowed=set(requested),
        )
        resolved_limits = resolve_benchmark_limits(
            requested,
            default_limit=args.limit,
            overrides=limit_overrides,
            full=args.full,
        )
    except ValueError as error:
        parser.error(str(error))

    examples = load_selected_benchmarks(
        requested,
        limit=args.limit,
        seed=args.seed,
        limits_by_benchmark=resolved_limits,
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
    output_caps = {
        benchmark: cap
        for benchmark, cap in {
            "musique": args.musique_max_output_tokens,
            "math": args.math_max_output_tokens,
            "aime2024": args.aime_max_output_tokens,
            "aime2025": args.aime_max_output_tokens,
            "mbpp_plus": args.mbpp_max_output_tokens,
        }.items()
        if cap is not None
    }
    report = run_factorial_experiment(
        examples,
        CoScopeSettings.from_env(".env"),
        reasoning_modes=reasoning_modes,
        sharing_arms=sharing_arms,
        batch_modes=batch_modes,
        threshold=args.threshold,
        max_output_tokens=args.max_output_tokens,
        max_output_tokens_by_benchmark=output_caps,
        bootstrap_samples=args.bootstrap_samples,
        code_evaluator=code_evaluator,
        progress=lambda event: print(
            json.dumps({"progress": event}, ensure_ascii=False),
            flush=True,
        ),
    )
    examples_total = sum(len(items) for items in examples.values())
    calls_per_example = len(sharing_arms) * len(batch_modes) * len(AGENTS) * sum(
        FACTORIAL_MODE_SPECS[mode].llm_calls_per_agent
        for mode in reasoning_modes
    )
    report["run_config"] = {
        "full": args.full,
        "seed": args.seed,
        "threshold": args.threshold,
        "benchmarks": requested,
        "limits_by_benchmark": resolved_limits,
        "reasoning_modes": [mode.value for mode in reasoning_modes],
        "sharing_policies": [arm.value for arm in sharing_arms],
        "batch_modes": [mode.value for mode in batch_modes],
        "factorial_conditions": (
            len(reasoning_modes) * len(sharing_arms) * len(batch_modes)
        ),
        "examples_total": examples_total,
        "planned_llm_calls": examples_total * calls_per_example,
        "max_output_tokens": args.max_output_tokens,
        "benchmark_output_caps": output_caps,
        "evalplus_image": (
            args.evalplus_image if "mbpp_plus" in requested else None
        ),
        "purpose": "reasoning_x_sharing_x_batch_factorial",
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
                "run_config": report["run_config"],
                "overall": report["overall"],
                "comparisons_by_mode": report["comparisons_by_mode"],
                "mode_comparisons_by_condition": report[
                    "mode_comparisons_by_condition"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
