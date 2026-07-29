"""Validate full benchmark coverage and estimate experiment work without model calls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from coscope.evaluation.benchmark_registry import (
    BENCHMARK_LOADERS,
    FULL_BENCHMARK_SIZES,
    benchmark_path,
    load_selected_benchmarks,
    parse_benchmark_limits,
    resolve_benchmark_limits,
)
from coscope.evaluation.benchmarks import BenchmarkExample
from coscope.evaluation.factorial_experiment import (
    DEFAULT_FACTORIAL_MODES,
    DEFAULT_SHARING_ARMS,
    FACTORIAL_MODE_SPECS,
    parse_reasoning_modes,
    parse_sharing_arms,
)
from coscope.evaluation.mas_experiment import AGENTS
from coscope.evaluation.sharing_ablation import SharingArm
from coscope.reasoning import ReasoningMode

AGENT_IDS = tuple(agent_id for agent_id, _, _ in AGENTS)

LLM_CALLS_PER_EXAMPLE = {
    "suite": 3,
    "sharing_ablation": 9,
    "reasoning_modes": 5,
    "factorial": 45,
}
TASK_RECORDS_PER_EXAMPLE = {
    "suite": 1,
    "sharing_ablation": 3,
    "reasoning_modes": 2,
    "factorial": 6,
}


def build_preflight_report(
    examples_by_benchmark: dict[str, list[BenchmarkExample]],
    *,
    requested_limits: dict[str, int],
    workflow: str,
    reasoning_modes: tuple[ReasoningMode, ...] = DEFAULT_FACTORIAL_MODES,
    sharing_arms: tuple[SharingArm, ...] = DEFAULT_SHARING_ARMS,
) -> dict[str, Any]:
    """Validate loaded examples and summarize the planned experiment."""
    errors: list[str] = []
    seen_ids: set[tuple[str, str]] = set()
    benchmark_reports: dict[str, dict[str, Any]] = {}

    for benchmark, examples in examples_by_benchmark.items():
        duplicate_ids: list[str] = []
        malformed = 0
        context_source_duplicates = 0
        for example in examples:
            key = (benchmark, example.example_id)
            if key in seen_ids:
                duplicate_ids.append(example.example_id)
            seen_ids.add(key)
            if (
                example.benchmark != benchmark
                or not example.example_id.strip()
                or not example.question.strip()
                or not example.reference.strip()
            ):
                malformed += 1
            source_ids = [item.source_id for item in example.context]
            if len(source_ids) != len(set(source_ids)):
                context_source_duplicates += 1
            if benchmark == "mbpp_plus":
                if set(example.metadata) != {
                    "task_id",
                    "entry_point",
                    "dataset_path",
                }:
                    errors.append(
                        f"{example.example_id}: MBPP metadata exposes unexpected fields"
                    )
                if example.reference != "[hidden EvalPlus base and plus tests]":
                    errors.append(
                        f"{example.example_id}: MBPP hidden reference marker changed"
                    )

        requested = requested_limits[benchmark]
        if len(examples) != requested:
            errors.append(
                f"{benchmark}: requested {requested} examples but loaded {len(examples)}"
            )
        if duplicate_ids:
            errors.append(f"{benchmark}: duplicate example IDs: {duplicate_ids[:5]}")
        if malformed:
            errors.append(f"{benchmark}: {malformed} malformed examples")
        if context_source_duplicates:
            errors.append(
                f"{benchmark}: {context_source_duplicates} examples have duplicate "
                "context source IDs"
            )
        benchmark_reports[benchmark] = {
            "requested_examples": requested,
            "loaded_examples": len(examples),
            "unique_example_ids": len({item.example_id for item in examples}),
            "examples_with_context": sum(bool(item.context) for item in examples),
            "context_items": sum(len(item.context) for item in examples),
            "reference_visibility": (
                "hidden_external_scorer"
                if benchmark == "mbpp_plus"
                else "scoring_only"
            ),
        }

    total_examples = sum(len(items) for items in examples_by_benchmark.values())
    if workflow == "factorial":
        calls_per_example = len(sharing_arms) * len(AGENT_IDS) * sum(
            FACTORIAL_MODE_SPECS[mode].llm_calls_per_agent
            for mode in reasoning_modes
        )
        records_per_example = len(reasoning_modes) * len(sharing_arms)
        evalplus_runs_per_benchmark = records_per_example
    else:
        calls_per_example = LLM_CALLS_PER_EXAMPLE[workflow]
        records_per_example = TASK_RECORDS_PER_EXAMPLE[workflow]
        evalplus_runs_per_benchmark = records_per_example
    mbpp_selected = "mbpp_plus" in examples_by_benchmark
    evalplus_runs = (
        evalplus_runs_per_benchmark if mbpp_selected else 0
    )
    planned: dict[str, Any] = {
        "benchmarks": list(examples_by_benchmark),
        "total_examples": total_examples,
        "llm_calls": total_examples * calls_per_example,
        "task_records": total_examples * records_per_example,
        "evalplus_container_runs": evalplus_runs,
    }
    if workflow == "factorial":
        planned.update(
            {
                "reasoning_modes": [mode.value for mode in reasoning_modes],
                "sharing_policies": [arm.value for arm in sharing_arms],
                "factorial_conditions": len(reasoning_modes) * len(sharing_arms),
                "retrieval_requests": (
                    total_examples
                    * len(sharing_arms)
                    * len(AGENT_IDS)
                    * sum(
                        len(
                            FACTORIAL_MODE_SPECS[mode].retrieval_operations
                        )
                        for mode in reasoning_modes
                    )
                ),
            }
        )
    return {
        "status": "ready" if not errors else "failed",
        "workflow": workflow,
        "model_calls_per_example": calls_per_example,
        "planned": planned,
        "metric_families": [
            "official task success (EM/F1, AIME accuracy, MBPP base/plus pass@1)",
            "LLM prompt/completion/reasoning/cached/total tokens",
            "embedding calls and input tokens",
            "retrieval grouping, savings, Recall@K, MRR@K, fallback",
            "context duplication and unauthorized exposure",
            "retrieval, generation, and end-to-end latency",
        ],
        "benchmarks": benchmark_reports,
        "errors": errors,
    }


def _data_inventory(
    names: list[str],
    *,
    data_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for name in names:
        path = benchmark_path(name, data_root=data_root)
        files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
        inventory[name] = {
            "path": str(path),
            "exists": path.exists(),
            "bytes": sum(item.stat().st_size for item in files if item.exists()),
            "files": [str(item) for item in files],
        }
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        default=",".join(BENCHMARK_LOADERS),
        help="comma-separated benchmark names",
    )
    parser.add_argument("--limit", type=int, default=5)
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
        "--workflow",
        choices=sorted(LLM_CALLS_PER_EXAMPLE),
        default="sharing_ablation",
    )
    parser.add_argument("--reasoning-modes", default="cot,tot")
    parser.add_argument(
        "--sharing-policies",
        default="coscope,full_sharing,no_sharing",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("COSCOPE_DATA_ROOT", "raw"),
    )
    parser.add_argument(
        "--output",
        default="new_results/full_experiment_preflight.json",
    )
    args = parser.parse_args()

    requested = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    if not requested:
        parser.error("--benchmarks must contain at least one benchmark")
    unknown = sorted(set(requested) - BENCHMARK_LOADERS.keys())
    if unknown:
        parser.error(
            f"unknown benchmarks: {', '.join(unknown)}; "
            f"choose from {', '.join(BENCHMARK_LOADERS)}"
        )
    try:
        reasoning_modes = parse_reasoning_modes(args.reasoning_modes)
        sharing_arms = parse_sharing_arms(args.sharing_policies)
        overrides = parse_benchmark_limits(
            args.benchmark_limits,
            allowed=set(requested),
        )
        limits = resolve_benchmark_limits(
            requested,
            default_limit=args.limit,
            overrides=overrides,
            full=args.full,
        )
    except ValueError as error:
        parser.error(str(error))

    inventory = _data_inventory(requested, data_root=args.data_root)
    missing = [name for name, item in inventory.items() if not item["exists"]]
    if missing:
        parser.error(f"missing local datasets: {', '.join(missing)}")
    examples = load_selected_benchmarks(
        requested,
        limit=args.limit,
        seed=args.seed,
        limits_by_benchmark=limits,
        data_root=args.data_root,
    )
    report = build_preflight_report(
        examples,
        requested_limits=limits,
        workflow=args.workflow,
        reasoning_modes=reasoning_modes,
        sharing_arms=sharing_arms,
    )
    report["run_config"] = {
        "full": args.full,
        "seed": args.seed,
        "data_root": str(Path(args.data_root).expanduser()),
        "limits_by_benchmark": limits,
        "known_full_split_sizes": {
            name: FULL_BENCHMARK_SIZES[name] for name in requested
        },
    }
    report["data_inventory"] = inventory

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **report}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
