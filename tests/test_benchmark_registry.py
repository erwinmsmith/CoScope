from __future__ import annotations

import json
from pathlib import Path

import pytest

from coscope.evaluation.benchmark_registry import (
    benchmark_path,
    parse_benchmark_limits,
    resolve_benchmark_limits,
)
from coscope.evaluation.benchmarks import BenchmarkExample, load_2wikimultihopqa
from coscope.scripts.preflight_experiment import build_preflight_report


def test_parse_and_resolve_per_benchmark_limits() -> None:
    overrides = parse_benchmark_limits(
        "aime2024=30,mbpp_plus=378",
        allowed={"aime2024", "mbpp_plus"},
    )

    assert resolve_benchmark_limits(
        ["aime2024", "mbpp_plus"],
        default_limit=5,
        overrides=overrides,
    ) == {"aime2024": 30, "mbpp_plus": 378}


@pytest.mark.parametrize(
    "value",
    ["aime2024", "aime2024=0", "aime2024=bad", "aime2024=1,aime2024=2"],
)
def test_reject_invalid_benchmark_limits(value: str) -> None:
    with pytest.raises(ValueError):
        parse_benchmark_limits(value)


def test_full_limits_use_known_split_sizes() -> None:
    assert resolve_benchmark_limits(
        ["aime2024", "aime2025", "mbpp_plus"],
        default_limit=1,
        full=True,
    ) == {"aime2024": 30, "aime2025": 30, "mbpp_plus": 378}


def test_benchmark_path_can_live_outside_the_repository(tmp_path: Path) -> None:
    assert benchmark_path("gsm8k", data_root=tmp_path) == (
        tmp_path / "gsm8k/test.parquet"
    )


def test_preflight_estimates_sharing_work_and_checks_hidden_code_reference() -> None:
    examples = {
        "aime2024": [
            BenchmarkExample(
                "aime_2024_I_01",
                "problem",
                "123",
                benchmark="aime2024",
            )
        ],
        "mbpp_plus": [
            BenchmarkExample(
                "mbpp_2",
                "Write a function.",
                "[hidden EvalPlus base and plus tests]",
                benchmark="mbpp_plus",
                metadata={
                    "task_id": "Mbpp/2",
                    "entry_point": "similar_elements",
                    "dataset_path": "/tmp/MbppPlus.jsonl.gz",
                },
            )
        ],
    }

    report = build_preflight_report(
        examples,
        requested_limits={"aime2024": 1, "mbpp_plus": 1},
        workflow="sharing_ablation",
    )

    assert report["status"] == "ready"
    assert report["planned"] == {
        "benchmarks": ["aime2024", "mbpp_plus"],
        "total_examples": 2,
        "llm_calls": 18,
        "task_records": 6,
        "evalplus_container_runs": 3,
    }


def test_preflight_estimates_default_factorial_matrix() -> None:
    examples = {
        "mbpp_plus": [
            BenchmarkExample(
                "mbpp_2",
                "Write a function.",
                "[hidden EvalPlus base and plus tests]",
                benchmark="mbpp_plus",
                metadata={
                    "task_id": "Mbpp/2",
                    "entry_point": "similar_elements",
                    "dataset_path": "/tmp/MbppPlus.jsonl.gz",
                },
            )
        ]
    }

    report = build_preflight_report(
        examples,
        requested_limits={"mbpp_plus": 1},
        workflow="factorial",
    )

    assert report["model_calls_per_example"] == 45
    assert report["planned"]["factorial_conditions"] == 6
    assert report["planned"]["retrieval_requests"] == 36
    assert report["planned"]["llm_calls"] == 45
    assert report["planned"]["task_records"] == 6
    assert report["planned"]["evalplus_container_runs"] == 6


def test_2wiki_loader_deduplicates_identical_titled_context(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps(
            [
                {
                    "_id": "example",
                    "question": "Question?",
                    "answer": "Answer",
                    "type": "comparison",
                    "context": [
                        ["Duplicate", ["Same paragraph."]],
                        ["Evidence", ["Useful paragraph."]],
                        ["Duplicate", ["Same paragraph."]],
                    ],
                    "supporting_facts": [["Evidence", 0]],
                }
            ]
        ),
        encoding="utf-8",
    )

    example = load_2wikimultihopqa(path, limit=1, seed=1)[0]

    assert [item.source_id for item in example.context] == [
        "Duplicate",
        "Evidence",
    ]
    assert example.supporting_source_ids == frozenset({"Evidence"})
