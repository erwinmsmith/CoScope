"""Run the factorial experiment with durable checkpoints and parallel workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from coscope.adapters import build_embedding
from coscope.config import CoScopeSettings
from coscope.evaluation.benchmark_registry import (
    BENCHMARK_LOADERS,
    benchmark_path,
    load_selected_benchmarks,
    parse_benchmark_limits,
    resolve_benchmark_limits,
)
from coscope.evaluation.benchmarks import BenchmarkExample
from coscope.evaluation.checkpoint_store import (
    ActiveCoordinatorError,
    ClaimedEvalJob,
    ClaimedTask,
    ExperimentCheckpoint,
    ExperimentTask,
)
from coscope.evaluation.code_benchmark import (
    DEFAULT_EVALPLUS_IMAGE,
    EvalPlusDockerEvaluator,
)
from coscope.evaluation.factorial_experiment import (
    DEFAULT_SHARING_ARMS,
    BatchMode,
    _factorial_aggregate,
    _mode_comparisons,
    parse_batch_modes,
    parse_reasoning_modes,
    parse_sharing_arms,
    run_factorial_task,
)
from coscope.evaluation.sharing_ablation import (
    SharingArm,
    _comparisons,
)
from coscope.memory import build_qdrant_store
from coscope.reasoning import ReasoningMode


class StopController:
    def __init__(self) -> None:
        self.requested = threading.Event()

    def install(self) -> None:
        def request_stop(signum: int, frame: object) -> None:
            del frame
            print(
                json.dumps(
                    {
                        "event": "stop_requested",
                        "signal": signum,
                        "message": "no new tasks will be scheduled",
                    }
                ),
                flush=True,
            )
            self.requested.set()

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    requested = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    unknown = sorted(set(requested) - BENCHMARK_LOADERS.keys())
    if unknown:
        parser.error(f"unknown benchmarks: {', '.join(unknown)}")
    try:
        modes = parse_reasoning_modes(args.reasoning_modes)
        policies = parse_sharing_arms(args.sharing_policies)
        batch_modes = parse_batch_modes(args.batch_modes)
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

    code_revision, is_dirty = _code_revision()
    if is_dirty and not args.allow_dirty_code:
        parser.error(
            "cloud runs require a clean Git checkout; commit/push the code first "
            "or use --allow-dirty-code only for local testing"
        )

    examples = load_selected_benchmarks(
        requested,
        limit=args.limit,
        seed=args.seed,
        limits_by_benchmark=limits,
        data_root=args.data_root,
    )
    examples_by_id = {
        (benchmark, example.example_id): example
        for benchmark, items in examples.items()
        for example in items
    }
    tasks = [
        ExperimentTask.create(
            benchmark,
            example.example_id,
            mode.value,
            policy.value,
            batch_mode.value,
        )
        for benchmark, items in examples.items()
        for example in items
        for mode in modes
        for policy in policies
        for batch_mode in batch_modes
    ]
    output_caps = _output_caps(args)
    settings = CoScopeSettings.from_env(args.env_file)
    if settings.runtime_mode != "live":
        parser.error("cloud factorial runs require COSCOPE_RUNTIME_MODE=live")
    settings.llm.validate()
    settings.embedding.validate()
    settings.retrieval.validate()
    settings.memory.validate()
    if settings.memory.provider != "qdrant":
        parser.error(
            "cloud factorial runs require "
            "COSCOPE_MEMORY_PROVIDER=qdrant"
        )
    embedding = build_embedding(settings.embedding)
    embedding_model_version = embedding.model_version
    qdrant_probe = build_qdrant_store(
        settings.memory,
        dimension=settings.embedding.dimension,
        namespace=f"preflight_{uuid.uuid4().hex}",
    )
    qdrant_probe.clear()
    state_dir = Path(args.state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 2,
        "purpose": "reasoning_x_sharing_x_batch_factorial",
        "code_revision": code_revision,
        "benchmarks": requested,
        "limits_by_benchmark": limits,
        "dataset_digest": _dataset_digest(examples),
        "reasoning_modes": [mode.value for mode in modes],
        "sharing_policies": [policy.value for policy in policies],
        "batch_modes": [batch_mode.value for batch_mode in batch_modes],
        "seed": args.seed,
        "threshold": args.threshold,
        "max_output_tokens": args.max_output_tokens,
        "benchmark_output_caps": output_caps,
        "llm_provider": settings.llm.provider,
        "llm_model": settings.llm.model,
        "llm_temperature": settings.llm.temperature,
        "embedding_provider": settings.embedding.provider,
        "embedding_model": settings.embedding.model,
        "embedding_dimension": settings.embedding.dimension,
        "embedding_model_version": embedding_model_version,
        "embedding_threads": settings.embedding.threads,
        "embedding_batch_size": settings.embedding.batch_size,
        "embedding_result_cache_size": (
            settings.embedding.result_cache_size
        ),
        "memory_provider": settings.memory.provider,
        "qdrant_collection": settings.memory.qdrant_collection,
        "bootstrap_samples": args.bootstrap_samples,
        "purge_details_after_success": args.purge_details_after_success,
        "canonical_latency_metric": (
            "wall_clock_seconds_minus_embedding_seconds"
        ),
    }
    completed_path = state_dir / "COMPLETED.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("config") != config:
            parser.error(
                "state directory contains a completed experiment with a "
                "different manifest; use a new state directory"
            )
        print(json.dumps(completed, ensure_ascii=False, indent=2))
        return 0
    checkpoint = ExperimentCheckpoint(state_dir / "checkpoint.sqlite3")
    fingerprint = checkpoint.initialize_manifest(
        config,
        tasks,
        include_mbpp_eval="mbpp_plus" in requested,
        allow_code_revision_change=args.allow_code_revision_change,
    )
    owner_id = f"{os.uname().nodename}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    try:
        recovered = checkpoint.acquire_coordinator(
            owner_id,
            stale_after_seconds=args.coordinator_timeout,
        )
    except ActiveCoordinatorError as error:
        parser.error(str(error))
    if args.retry_final_failures:
        reset_failures = checkpoint.retry_final_failures()
    else:
        reset_failures = 0

    stop = StopController()
    stop.install()
    event_path = state_dir / "events.jsonl"
    _append_event(
        event_path,
        {
            "event": "coordinator_started",
            "owner_id": owner_id,
            "fingerprint": fingerprint,
            "workers": args.workers,
            "recovered_running_tasks": recovered,
            "reset_failures": reset_failures,
        },
    )
    _write_json_atomic(state_dir / "manifest.json", config)
    _write_status(checkpoint, state_dir, owner_id=owner_id)

    futures: dict[Future[dict[str, Any]], ClaimedTask] = {}
    exit_code = 0
    last_heartbeat = 0.0
    try:
        with ThreadPoolExecutor(
            max_workers=args.workers,
            thread_name_prefix="coscope",
        ) as executor:
            while futures or not stop.requested.is_set():
                open_slots = args.workers - len(futures)
                if open_slots > 0 and not stop.requested.is_set():
                    claimed = checkpoint.claim_tasks(
                        owner_id,
                        limit=open_slots,
                        max_attempts=args.max_attempts,
                    )
                    for task in claimed:
                        future = executor.submit(
                            _execute_task,
                            task,
                            examples_by_id,
                            settings,
                            args.threshold,
                            args.max_output_tokens,
                            output_caps,
                        )
                        futures[future] = task
                        _append_event(
                            event_path,
                            _task_event("task_started", task),
                        )

                if not futures:
                    counts = checkpoint.counts()
                    if counts["pending"] == 0:
                        break
                    time.sleep(checkpoint.next_ready_delay())
                else:
                    completed, _ = wait(
                        tuple(futures),
                        timeout=min(args.status_interval, 5.0),
                    )
                    for future in completed:
                        task = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as error:
                            redacted = _redact_error(error, settings)
                            status = checkpoint.fail_task(
                                owner_id,
                                task.task_key,
                                redacted,
                                max_attempts=args.max_attempts,
                                retry_backoff_seconds=args.retry_backoff_seconds,
                            )
                            _append_event(
                                event_path,
                                {
                                    **_task_event("task_failed", task),
                                    "status": status,
                                    "error": redacted,
                                },
                            )
                        else:
                            checkpoint.complete_task(
                                owner_id,
                                task.task_key,
                                result,
                                retain_details=(
                                    not args.purge_details_after_success
                                    or task.benchmark == "mbpp_plus"
                                ),
                            )
                            _append_event(
                                event_path,
                                {
                                    **_task_event("task_succeeded", task),
                                    "metrics": _compact_metrics(result),
                                },
                            )
                        _write_status(
                            checkpoint,
                            state_dir,
                            owner_id=owner_id,
                        )

                now = time.monotonic()
                if now - last_heartbeat >= args.status_interval:
                    checkpoint.heartbeat(owner_id)
                    _write_status(
                        checkpoint,
                        state_dir,
                        owner_id=owner_id,
                    )
                    last_heartbeat = now

        counts = checkpoint.counts()
        if not stop.requested.is_set() and counts["failed"] == 0:
            eval_ok = _run_evalplus_jobs(
                checkpoint,
                owner_id,
                examples,
                args,
                state_dir,
                event_path,
                settings,
            )
            if not eval_ok:
                exit_code = 2
        elif counts["failed"]:
            exit_code = 2
        elif stop.requested.is_set():
            exit_code = 130

        if exit_code == 0 and args.export_results:
            exported = checkpoint.export_results_jsonl(
                state_dir / "results.jsonl"
            )
            _append_event(
                event_path,
                {"event": "results_exported", "records": exported},
            )
        if exit_code == 0:
            final_metrics = _build_final_metrics(
                checkpoint,
                config,
                modes,
                policies,
                batch_modes,
                bootstrap_samples=args.bootstrap_samples,
            )
            _write_json_atomic(
                state_dir / "final_metrics.json",
                final_metrics,
            )
            _write_json_atomic(
                completed_path,
                {
                    "status": "completed",
                    "fingerprint": fingerprint,
                    "completed_at": time.time(),
                    "config": config,
                    "final_metrics": str(
                        state_dir / "final_metrics.json"
                    ),
                },
            )
        final_snapshot = checkpoint.status_snapshot()
    finally:
        _write_status(
            checkpoint,
            state_dir,
            owner_id=owner_id,
            final=True,
        )
        checkpoint.release_coordinator(owner_id)
        _append_event(
            event_path,
            {
                "event": "coordinator_stopped",
                "owner_id": owner_id,
                "exit_code": exit_code,
            },
        )

    if exit_code == 0 and args.purge_details_after_success:
        _purge_intermediate_files(state_dir)

    print(
        json.dumps(
            {
                "state_dir": str(state_dir),
                "fingerprint": fingerprint,
                "exit_code": exit_code,
                "status": final_snapshot,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", default=",".join(BENCHMARK_LOADERS))
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--benchmark-limits", default="")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--reasoning-modes", default="cot,tot")
    parser.add_argument(
        "--sharing-policies",
        default="coscope,full_sharing,no_sharing",
    )
    parser.add_argument(
        "--batch-modes",
        default="batched,independent",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--musique-max-output-tokens", type=int)
    parser.add_argument("--math-max-output-tokens", type=int)
    parser.add_argument("--aime-max-output-tokens", type=int)
    parser.add_argument("--mbpp-max-output-tokens", type=int)
    parser.add_argument(
        "--env-file",
        default=os.environ.get("COSCOPE_ENV_FILE", ".env"),
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("COSCOPE_DATA_ROOT", "raw"),
    )
    parser.add_argument(
        "--state-dir",
        default=os.environ.get(
            "COSCOPE_RUN_ROOT",
            "experiment_runs/factorial",
        ),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--retry-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--retry-final-failures", action="store_true")
    parser.add_argument("--status-interval", type=float, default=15.0)
    parser.add_argument("--coordinator-timeout", type=float, default=120.0)
    parser.add_argument("--evalplus-image", default=DEFAULT_EVALPLUS_IMAGE)
    parser.add_argument("--evalplus-artifact-dir")
    parser.add_argument("--evalplus-parallel", type=int, default=2)
    parser.add_argument("--evalplus-jobs", type=int, default=1)
    parser.add_argument(
        "--evalplus-memory",
        default=os.environ.get("COSCOPE_EVALPLUS_MEMORY", "4g"),
    )
    parser.add_argument("--export-results", action="store_true")
    parser.add_argument(
        "--purge-details-after-success",
        action="store_true",
        help=(
            "retain compact metrics during execution, then remove checkpoint "
            "details after final_metrics.json is committed"
        ),
    )
    parser.add_argument("--allow-dirty-code", action="store_true")
    parser.add_argument(
        "--allow-code-revision-change",
        action="store_true",
        help=(
            "resume only when the Git revision is the sole manifest change; "
            "records the revision transition in checkpoint metadata"
        ),
    )
    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.full and not args.confirm_full_run:
        parser.error("--full requires --confirm-full-run")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    for name in (
        "workers",
        "max_attempts",
        "bootstrap_samples",
        "status_interval",
        "coordinator_timeout",
        "evalplus_parallel",
        "evalplus_jobs",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.retry_backoff_seconds < 0:
        parser.error("--retry-backoff-seconds cannot be negative")
    if (
        re.fullmatch(
            r"[1-9]\d*[kmgt](?:i?b)?",
            args.evalplus_memory.casefold(),
        )
        is None
    ):
        parser.error("--evalplus-memory must be a Docker size such as 2g")
    for cap in (
        args.max_output_tokens,
        args.musique_max_output_tokens,
        args.math_max_output_tokens,
        args.aime_max_output_tokens,
        args.mbpp_max_output_tokens,
    ):
        if cap is not None and cap <= 0:
            parser.error("output token limits must be positive")


def _execute_task(
    task: ClaimedTask,
    examples_by_id: dict[tuple[str, str], BenchmarkExample],
    settings: CoScopeSettings,
    threshold: float,
    default_cap: int | None,
    output_caps: dict[str, int],
) -> dict[str, Any]:
    example = examples_by_id[(task.benchmark, task.example_id)]
    cap = output_caps.get(task.benchmark, default_cap)
    return run_factorial_task(
        example,
        settings,
        ReasoningMode(task.reasoning_mode),
        SharingArm(task.sharing_policy),
        BatchMode(task.batch_mode),
        threshold=threshold,
        max_output_tokens=cap,
    )


def _run_evalplus_jobs(
    checkpoint: ExperimentCheckpoint,
    owner_id: str,
    examples: dict[str, list[BenchmarkExample]],
    args: argparse.Namespace,
    state_dir: Path,
    event_path: Path,
    settings: CoScopeSettings,
) -> bool:
    if "mbpp_plus" not in examples:
        return True
    artifact_dir = Path(
        args.evalplus_artifact_dir
        or state_dir / "evalplus"
    )
    evaluator = EvalPlusDockerEvaluator(
        benchmark_path("mbpp_plus", data_root=args.data_root),
        artifact_root=artifact_dir,
        image=args.evalplus_image,
        parallel=args.evalplus_parallel,
        memory=args.evalplus_memory,
    )
    example_list = examples["mbpp_plus"]
    futures: dict[Future[dict[str, Any]], ClaimedEvalJob] = {}
    with ThreadPoolExecutor(
        max_workers=args.evalplus_jobs,
        thread_name_prefix="evalplus",
    ) as executor:
        while futures or checkpoint.eval_counts()["pending"]:
            open_slots = args.evalplus_jobs - len(futures)
            for job in checkpoint.claim_eval_jobs(
                owner_id,
                limit=open_slots,
                max_attempts=args.max_attempts,
            ):
                predictions = {
                    result["example_id"]: result["prediction"]
                    for result in checkpoint.iter_results(
                        benchmark="mbpp_plus",
                        reasoning_mode=job.reasoning_mode,
                        sharing_policy=job.sharing_policy,
                        batch_mode=job.batch_mode,
                    )
                }
                future = executor.submit(
                    evaluator.evaluate,
                    example_list,
                    predictions,
                    label=job.job_key,
                )
                futures[future] = job
                _append_event(
                    event_path,
                    {
                        "event": "evalplus_started",
                        "job_key": job.job_key,
                        "attempt": job.attempt,
                    },
                )
            if not futures:
                time.sleep(1)
                continue
            completed, _ = wait(tuple(futures), timeout=5)
            for future in completed:
                job = futures.pop(future)
                try:
                    scores = future.result()
                    checkpoint.complete_eval_job(
                        owner_id,
                        job,
                        scores,
                    )
                except Exception as error:
                    redacted = _redact_error(error, settings)
                    status = checkpoint.fail_eval_job(
                        owner_id,
                        job,
                        redacted,
                        max_attempts=args.max_attempts,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                    )
                    _append_event(
                        event_path,
                        {
                            "event": "evalplus_failed",
                            "job_key": job.job_key,
                            "status": status,
                            "error": redacted,
                        },
                    )
                else:
                    _append_event(
                        event_path,
                        {
                            "event": "evalplus_succeeded",
                            "job_key": job.job_key,
                            "scored_examples": len(scores),
                        },
                    )
                checkpoint.heartbeat(owner_id)
                _write_status(checkpoint, state_dir, owner_id=owner_id)
    return checkpoint.eval_counts()["failed"] == 0


def _output_caps(args: argparse.Namespace) -> dict[str, int]:
    return {
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


def _dataset_digest(
    examples: dict[str, list[BenchmarkExample]],
) -> str:
    digest = hashlib.sha256()
    for benchmark, items in examples.items():
        for example in items:
            payload = {
                "benchmark": benchmark,
                "example_id": example.example_id,
                "question": example.question,
                "reference": example.reference,
                "context": [
                    (context.source_id, context.content)
                    for context in example.context
                ],
                "metadata": example.metadata,
            }
            digest.update(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
    return digest.hexdigest()


def _build_final_metrics(
    checkpoint: ExperimentCheckpoint,
    config: dict[str, Any],
    modes: tuple[ReasoningMode, ...],
    policies: tuple[SharingArm, ...],
    batch_modes: tuple[BatchMode, ...],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    benchmark_reports: dict[str, Any] = {}
    for benchmark in config["benchmarks"]:
        benchmark_reports[benchmark] = {
            mode.value: {
                policy.value: {
                    batch_mode.value: _factorial_aggregate(
                        list(
                            checkpoint.iter_metric_records(
                                benchmark=benchmark,
                                reasoning_mode=mode.value,
                                sharing_policy=policy.value,
                                batch_mode=batch_mode.value,
                            )
                        )
                    )
                    for batch_mode in batch_modes
                }
                for policy in policies
            }
            for mode in modes
        }

    overall: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    comparisons_by_mode: dict[str, Any] = {}
    for mode in modes:
        overall[mode.value] = {
            policy.value: {
                batch_mode.value: _factorial_aggregate(
                    list(
                        checkpoint.iter_metric_records(
                            reasoning_mode=mode.value,
                            sharing_policy=policy.value,
                            batch_mode=batch_mode.value,
                        )
                    )
                )
                for batch_mode in batch_modes
            }
            for policy in policies
        }
        if set(policies) == set(DEFAULT_SHARING_ARMS):
            comparisons_by_mode[mode.value] = {}
            for batch_mode in batch_modes:
                records_by_policy = {
                    policy: list(
                        checkpoint.iter_metric_records(
                            reasoning_mode=mode.value,
                            sharing_policy=policy.value,
                            batch_mode=batch_mode.value,
                        )
                    )
                    for policy in policies
                }
                comparisons_by_mode[mode.value][batch_mode.value] = (
                    _comparisons(
                        records_by_policy,
                        {
                            policy.value: overall[mode.value][
                                policy.value
                            ][batch_mode.value]
                            for policy in policies
                        },
                        bootstrap_samples=bootstrap_samples,
                    )
                )

    return {
        "status": "completed",
        "completed_at": time.time(),
        "models": {
            "llm_provider": config["llm_provider"],
            "llm": config["llm_model"],
            "llm_temperature": config["llm_temperature"],
            "embedding_provider": config["embedding_provider"],
            "embedding": config["embedding_model"],
            "embedding_dimension": config["embedding_dimension"],
            "embedding_model_version": config[
                "embedding_model_version"
            ],
        },
        "design": {
            "factorial": True,
            "reasoning_modes": config["reasoning_modes"],
            "sharing_policies": config["sharing_policies"],
            "batch_modes": config["batch_modes"],
            "uniform_mode_across_agents": True,
            "intermediate_details_purged": config[
                "purge_details_after_success"
            ],
            "canonical_latency_metric": config[
                "canonical_latency_metric"
            ],
        },
        "run_config": config,
        "benchmarks": benchmark_reports,
        "overall": overall,
        "comparisons_by_mode": comparisons_by_mode,
        "mode_comparisons_by_condition": _mode_comparisons(
            overall,
            modes,
            policies,
            batch_modes,
        ),
    }


def _purge_intermediate_files(state_dir: Path) -> None:
    for name in (
        "checkpoint.sqlite3",
        "checkpoint.sqlite3-wal",
        "checkpoint.sqlite3-shm",
        "events.jsonl",
        "results.jsonl",
    ):
        candidate = state_dir / name
        if candidate.exists():
            candidate.unlink()


def _code_revision() -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return revision, dirty


def _task_event(event: str, task: ClaimedTask) -> dict[str, Any]:
    return {
        "event": event,
        "task_key": task.task_key,
        "benchmark": task.benchmark,
        "example_id": task.example_id,
        "reasoning_mode": task.reasoning_mode,
        "sharing_policy": task.sharing_policy,
        "batch_mode": task.batch_mode,
        "attempt": task.attempt,
    }


def _compact_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": result["task_success"],
        "answer_f1": result["scores"]["f1"],
        "llm_tokens": result["provider_usage"]
        .get("by_category", {})
        .get("llm", {})
        .get("total_tokens", 0),
        "provider_tokens": result["provider_usage"]["all"]["total_tokens"],
        "store_queries": result["retrieval"]["store_queries"],
        "query_reduction_rate": result["retrieval"][
            "query_reduction_rate"
        ],
        "end_to_end_seconds": result["latency"]["end_to_end_seconds"],
    }


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": time.time(), **event}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _write_status(
    checkpoint: ExperimentCheckpoint,
    state_dir: Path,
    *,
    owner_id: str,
    final: bool = False,
) -> None:
    snapshot = checkpoint.status_snapshot()
    snapshot["coordinator_owner"] = owner_id
    snapshot["final"] = final
    _write_json_atomic(state_dir / "status.json", snapshot)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _redact_error(
    error: BaseException,
    settings: CoScopeSettings,
) -> str:
    value = "".join(
        traceback.format_exception_only(type(error), error)
    ).strip()
    for secret in (
        settings.llm.api_key,
        settings.embedding.api_key,
        settings.memory.qdrant_api_key,
    ):
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value[:4_000]


if __name__ == "__main__":
    raise SystemExit(main())
