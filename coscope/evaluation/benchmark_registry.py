"""Canonical local paths and loaders for supported benchmark datasets."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path

from coscope.evaluation.benchmarks import (
    BenchmarkExample,
    load_2wikimultihopqa,
    load_aime2024,
    load_aime2025,
    load_gsm8k,
    load_hotpotqa,
    load_math,
    load_mbpp_plus,
    load_musique,
)

BenchmarkLoader = Callable[..., list[BenchmarkExample]]

BENCHMARK_LOADERS: dict[str, tuple[BenchmarkLoader, str]] = {
    "gsm8k": (load_gsm8k, "raw/gsm8k/test.parquet"),
    "hotpotqa": (
        load_hotpotqa,
        "raw/hotpotqa/distractor_validation.parquet",
    ),
    "2wikimultihopqa": (
        load_2wikimultihopqa,
        "raw/2wikimhqa/dev.json",
    ),
    "musique": (
        load_musique,
        "raw/musique/musique_ans_v1.0_dev.jsonl",
    ),
    "math": (load_math, "raw/math/test.parquet"),
    "aime2024": (load_aime2024, "raw/aime"),
    "aime2025": (load_aime2025, "raw/aime"),
    "mbpp_plus": (
        load_mbpp_plus,
        "raw/mbpp_plus/MbppPlus.jsonl.gz",
    ),
}

FULL_BENCHMARK_SIZES: dict[str, int] = {
    "gsm8k": 1_319,
    "hotpotqa": 7_405,
    "2wikimultihopqa": 12_576,
    "musique": 2_417,
    "math": 5_000,
    "aime2024": 30,
    "aime2025": 30,
    "mbpp_plus": 378,
}


def benchmark_path(
    name: str,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """Resolve a benchmark path under a configurable, non-repository data root."""
    if name not in BENCHMARK_LOADERS:
        raise ValueError(f"unknown benchmark: {name}")
    configured_root = data_root or os.environ.get("COSCOPE_DATA_ROOT", "raw")
    root = Path(configured_root).expanduser()
    relative = Path(BENCHMARK_LOADERS[name][1])
    if relative.parts and relative.parts[0] == "raw":
        relative = Path(*relative.parts[1:])
    return root / relative


def parse_benchmark_limits(
    value: str,
    *,
    allowed: set[str] | None = None,
) -> dict[str, int]:
    """Parse ``benchmark=count`` pairs from a comma-separated CLI value."""
    limits: dict[str, int] = {}
    if not value.strip():
        return limits
    for item in value.split(","):
        name, separator, raw_limit = item.strip().partition("=")
        if not separator or not name or not raw_limit:
            raise ValueError(
                "benchmark limits must use comma-separated benchmark=count pairs"
            )
        if allowed is not None and name not in allowed:
            raise ValueError(f"benchmark limit provided for unselected benchmark: {name}")
        try:
            limit = int(raw_limit)
        except ValueError as error:
            raise ValueError(f"benchmark limit must be an integer: {item}") from error
        if limit <= 0:
            raise ValueError(f"benchmark limit must be positive: {item}")
        if name in limits:
            raise ValueError(f"duplicate benchmark limit: {name}")
        limits[name] = limit
    return limits


def resolve_benchmark_limits(
    names: list[str],
    *,
    default_limit: int,
    overrides: Mapping[str, int] | None = None,
    full: bool = False,
) -> dict[str, int]:
    """Resolve one concrete sample count for each selected benchmark."""
    if default_limit <= 0:
        raise ValueError("default benchmark limit must be positive")
    unknown = set(names) - BENCHMARK_LOADERS.keys()
    if unknown:
        raise ValueError(f"unknown benchmarks: {', '.join(sorted(unknown))}")
    resolved = {
        name: FULL_BENCHMARK_SIZES[name] if full else default_limit for name in names
    }
    for name, limit in (overrides or {}).items():
        if name not in resolved:
            raise ValueError(f"benchmark limit provided for unselected benchmark: {name}")
        if limit <= 0:
            raise ValueError(f"benchmark limit must be positive: {name}={limit}")
        resolved[name] = limit
    return resolved


def load_selected_benchmarks(
    names: list[str],
    *,
    limit: int,
    seed: int,
    limits_by_benchmark: Mapping[str, int] | None = None,
    data_root: str | Path | None = None,
) -> dict[str, list[BenchmarkExample]]:
    limits = resolve_benchmark_limits(
        names,
        default_limit=limit,
        overrides=limits_by_benchmark,
    )
    return {
        name: BENCHMARK_LOADERS[name][0](
            benchmark_path(name, data_root=data_root),
            limit=limits[name],
            seed=seed,
        )
        for name in names
    }
