from __future__ import annotations

from pathlib import Path

import pytest

from coscope.evaluation.checkpoint_store import (
    ActiveCoordinatorError,
    CheckpointMismatchError,
    ExperimentCheckpoint,
    ExperimentTask,
)
from coscope.evaluation.code_benchmark import CodeScore


def _result() -> dict:
    return {
        "benchmark": "gsm8k",
        "example_id": "sample-1",
        "reasoning_mode": "cot",
        "arm": "coscope",
        "scores": {
            "em": 1.0,
            "f1": 1.0,
            "accuracy": 1.0,
            "success": 1.0,
        },
        "task_success": True,
        "provider_usage": {
            "all": {"calls": 2, "total_tokens": 21},
            "by_category": {
                "llm": {
                    "calls": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "reasoning_tokens": 2,
                    "cached_tokens": 0,
                    "total_tokens": 15,
                },
                "embedding": {
                    "calls": 1,
                    "prompt_tokens": 6,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 6,
                },
            },
        },
        "retrieval": {
            "requests": 3,
            "groups": 1,
            "store_queries": 1,
            "shared_store_queries": 1,
            "private_store_queries": 0,
            "shared_recall_savings": 2 / 3,
            "fallback_triggers": 0,
            "recall_at_10": 1.0,
            "mrr_at_10": 1.0,
            "latency_seconds": 0.5,
        },
        "context": {
            "selected_items": 3,
            "mean_duplicate_evidence_rate": 0.0,
            "polluted_items": 0,
            "pollution_rate": 0.0,
            "private_thinking_exposure": 0,
            "cross_role_knowledge_exposure": 0,
        },
        "safety": {"unauthorized_context_exposure": 0},
        "data_flow": {
            "planner": {
                "private_memory_id": "private",
                "published_memory_id": "public",
            }
        },
        "generation_budget": {"cap_hits": 0, "agents_at_cap": []},
        "latency": {
            "generation_seconds": 2.0,
            "end_to_end_seconds": 2.5,
        },
        "reasoning": {
            "planner": {
                "node_count": 1,
                "generated_thoughts": 1,
                "llm_calls": 1,
            }
        },
    }


def test_checkpoint_completion_is_not_claimed_twice(tmp_path: Path) -> None:
    store = ExperimentCheckpoint(tmp_path / "checkpoint.sqlite3")
    task = ExperimentTask.create("gsm8k", "sample-1", "cot", "coscope")
    store.initialize_manifest(
        {"seed": 1},
        [task],
        include_mbpp_eval=False,
    )
    assert store.acquire_coordinator("worker-a", stale_after_seconds=60) == 0
    claimed = store.claim_tasks("worker-a", limit=1, max_attempts=3)
    assert len(claimed) == 1
    store.complete_task("worker-a", task.task_key, _result())

    assert store.claim_tasks("worker-a", limit=1, max_attempts=3) == []
    assert store.counts() == {
        "pending": 0,
        "running": 0,
        "succeeded": 1,
        "failed": 0,
        "total": 1,
    }
    assert list(store.iter_results()) == [_result()]


def test_checkpoint_recovers_running_task_after_stale_owner(
    tmp_path: Path,
) -> None:
    store = ExperimentCheckpoint(tmp_path / "checkpoint.sqlite3")
    task = ExperimentTask.create("gsm8k", "sample-1", "tot", "no_sharing")
    store.initialize_manifest({}, [task], include_mbpp_eval=False)
    store.acquire_coordinator("worker-a", stale_after_seconds=60)
    first = store.claim_tasks("worker-a", limit=1, max_attempts=3)
    assert first[0].attempt == 1

    with pytest.raises(ActiveCoordinatorError):
        store.acquire_coordinator("worker-b", stale_after_seconds=60)
    recovered = store.acquire_coordinator(
        "worker-b",
        stale_after_seconds=0,
    )
    assert recovered == 1
    second = store.claim_tasks("worker-b", limit=1, max_attempts=3)
    assert second[0].task_key == task.task_key
    assert second[0].attempt == 2


def test_checkpoint_can_drop_details_but_keep_aggregate_inputs(
    tmp_path: Path,
) -> None:
    store = ExperimentCheckpoint(tmp_path / "checkpoint.sqlite3")
    task = ExperimentTask.create("gsm8k", "sample-1", "cot", "coscope")
    store.initialize_manifest({}, [task], include_mbpp_eval=False)
    store.acquire_coordinator("worker", stale_after_seconds=60)
    store.claim_tasks("worker", limit=1, max_attempts=2)
    store.complete_task(
        "worker",
        task.task_key,
        _result(),
        retain_details=False,
    )

    assert list(store.iter_results()) == []
    metrics = list(store.iter_metric_records())
    assert len(metrics) == 1
    assert metrics[0]["scores"]["f1"] == 1.0
    assert metrics[0]["provider_usage"]["by_category"]["llm"][
        "total_tokens"
    ] == 15


def test_checkpoint_rejects_changed_resume_manifest(tmp_path: Path) -> None:
    store = ExperimentCheckpoint(tmp_path / "checkpoint.sqlite3")
    task = ExperimentTask.create("gsm8k", "sample-1", "cot", "coscope")
    store.initialize_manifest(
        {"threshold": 0.75},
        [task],
        include_mbpp_eval=False,
    )
    with pytest.raises(CheckpointMismatchError):
        store.initialize_manifest(
            {"threshold": 0.90},
            [task],
            include_mbpp_eval=False,
        )


def test_checkpoint_failure_retries_then_stops(tmp_path: Path) -> None:
    store = ExperimentCheckpoint(tmp_path / "checkpoint.sqlite3")
    task = ExperimentTask.create("gsm8k", "sample-1", "cot", "coscope")
    store.initialize_manifest({}, [task], include_mbpp_eval=False)
    store.acquire_coordinator("worker", stale_after_seconds=60)

    for expected_attempt in (1, 2):
        claimed = store.claim_tasks("worker", limit=1, max_attempts=2)
        assert claimed[0].attempt == expected_attempt
        status = store.fail_task(
            "worker",
            task.task_key,
            "provider timeout",
            max_attempts=2,
            retry_backoff_seconds=0,
        )
    assert status == "failed"
    assert store.claim_tasks("worker", limit=1, max_attempts=2) == []

    assert store.retry_final_failures() == 1
    assert store.claim_tasks("worker", limit=1, max_attempts=2)[0].attempt == 1


def test_evalplus_scores_are_applied_atomically(tmp_path: Path) -> None:
    store = ExperimentCheckpoint(tmp_path / "checkpoint.sqlite3")
    task = ExperimentTask.create(
        "mbpp_plus",
        "mbpp-1",
        "cot",
        "coscope",
    )
    store.initialize_manifest({}, [task], include_mbpp_eval=True)
    store.acquire_coordinator("worker", stale_after_seconds=60)
    claimed = store.claim_tasks("worker", limit=1, max_attempts=2)[0]
    result = {
        **_result(),
        "benchmark": "mbpp_plus",
        "example_id": "mbpp-1",
        "prediction": "def answer():\n    return 1",
    }
    store.complete_task("worker", claimed.task_key, result)

    job = store.claim_eval_jobs("worker", limit=1, max_attempts=2)[0]
    assert store.complete_eval_job(
        "worker",
        job,
        {"mbpp-1": CodeScore(base_pass=True, plus_pass=False)},
    ) == 1

    evaluated = next(store.iter_results())
    assert evaluated["scores"]["base_pass"] == 1.0
    assert evaluated["scores"]["plus_pass"] == 0.0
    assert not evaluated["task_success"]
    assert store.eval_counts()["succeeded"] == 1
