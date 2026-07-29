"""Load benchmark examples without leaking gold answers into runtime context."""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

RowT = TypeVar("RowT")


@dataclass(frozen=True)
class BenchmarkContext:
    source_id: str
    content: str


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    question: str
    reference: str
    benchmark: str = "gsm8k"
    context: tuple[BenchmarkContext, ...] = ()
    aliases: tuple[str, ...] = ()
    supporting_source_ids: frozenset[str] = frozenset()
    metadata: dict[str, str] = field(default_factory=dict)


def load_gsm8k(
    path: str | Path,
    *,
    limit: int,
    seed: int,
) -> list[BenchmarkExample]:
    rows, indices = _sample_parquet(
        path,
        limit=limit,
        seed=seed,
        columns=["question", "answer"],
    )
    return [
        BenchmarkExample(f"gsm8k_test_{index}", row["question"], row["answer"])
        for index, row in zip(indices, rows, strict=True)
    ]


def load_math(
    path: str | Path,
    *,
    limit: int,
    seed: int,
) -> list[BenchmarkExample]:
    rows, indices = _sample_parquet(path, limit=limit, seed=seed)
    return [
        BenchmarkExample(
            f"math_test_{index}",
            row["problem"],
            row["solution"],
            benchmark="math",
            metadata={"level": row["level"], "type": row["type"]},
        )
        for index, row in zip(indices, rows, strict=True)
    ]


def load_aime(
    path: str | Path,
    *,
    year: int,
    limit: int,
    seed: int,
) -> list[BenchmarkExample]:
    """Load the 30 MathArena AIME I/II problems for 2024 or 2025."""
    if year not in {2024, 2025}:
        raise ValueError("AIME year must be 2024 or 2025")
    root = Path(path)
    if year == 2024:
        rows: list[tuple[str, int, dict]] = []
        for section in ("I", "II"):
            section_rows, _ = _read_parquet(
                root / f"aime_2024_{section}.parquet"
            )
            rows.extend(
                (section, int(row["problem_idx"]), row)
                for row in section_rows
            )
    else:
        year_rows, _ = _read_parquet(root / "aime_2025.parquet")
        rows = [
            (
                "I" if index < 15 else "II",
                index % 15 + 1,
                row,
            )
            for index, row in enumerate(year_rows)
        ]
    sampled, _ = _sample_rows(rows, limit=limit, seed=seed)
    return [
        BenchmarkExample(
            f"aime_{year}_{section}_{section_index:02d}",
            str(row["problem"]),
            str(row["answer"]),
            benchmark=f"aime{year}",
            metadata={
                "year": str(year),
                "section": section,
                "problem_idx": str(section_index),
                "source_problem_idx": str(row["problem_idx"]),
            },
        )
        for section, section_index, row in sampled
    ]


def load_aime2024(
    path: str | Path,
    *,
    limit: int,
    seed: int,
) -> list[BenchmarkExample]:
    return load_aime(path, year=2024, limit=limit, seed=seed)


def load_aime2025(
    path: str | Path,
    *,
    limit: int,
    seed: int,
) -> list[BenchmarkExample]:
    return load_aime(path, year=2025, limit=limit, seed=seed)


def load_mbpp_plus(
    path: str | Path,
    *,
    limit: int,
    seed: int,
) -> list[BenchmarkExample]:
    """Load EvalPlus MBPP+ prompts without exposing solutions or tests."""
    dataset_path = Path(path).resolve()
    with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    sampled, _ = _sample_rows(rows, limit=limit, seed=seed)
    return [
        BenchmarkExample(
            str(row["task_id"]).replace("/", "_").casefold(),
            str(row["prompt"]),
            "[hidden EvalPlus base and plus tests]",
            benchmark="mbpp_plus",
            metadata={
                "task_id": str(row["task_id"]),
                "entry_point": str(row["entry_point"]),
                "dataset_path": str(dataset_path),
            },
        )
        for row in sampled
    ]


def load_hotpotqa(
    path: str | Path,
    *,
    limit: int,
    seed: int,
) -> list[BenchmarkExample]:
    rows, _ = _sample_parquet(path, limit=limit, seed=seed)
    examples = []
    for row in rows:
        context = row["context"]
        items = tuple(
            BenchmarkContext(title, f"{title}\n{' '.join(sentences)}")
            for title, sentences in zip(
                context["title"], context["sentences"], strict=True
            )
        )
        examples.append(
            BenchmarkExample(
                row["id"],
                row["question"],
                row["answer"],
                benchmark="hotpotqa",
                context=items,
                supporting_source_ids=frozenset(row["supporting_facts"]["title"]),
                metadata={"type": row["type"], "level": row["level"]},
            )
        )
    return examples


def load_2wikimultihopqa(
    path: str | Path,
    *,
    limit: int,
    seed: int,
) -> list[BenchmarkExample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows, _ = _sample_rows(raw, limit=limit, seed=seed)
    return [
        BenchmarkExample(
            row["_id"],
            row["question"],
            row["answer"],
            benchmark="2wikimultihopqa",
            context=tuple(
                BenchmarkContext(title, f"{title}\n{' '.join(sentences)}")
                for title, sentences in _deduplicate_titled_context(row["context"])
            ),
            supporting_source_ids=frozenset(
                title for title, _ in row["supporting_facts"]
            ),
            metadata={"type": row["type"]},
        )
        for row in rows
    ]


def _deduplicate_titled_context(
    context: list[list[object]],
) -> list[tuple[str, list[str]]]:
    """Keep the first copy of exact duplicate titled paragraphs."""
    deduplicated: list[tuple[str, list[str]]] = []
    content_by_title: dict[str, list[str]] = {}
    for raw_title, raw_sentences in context:
        title = str(raw_title)
        if not isinstance(raw_sentences, list):
            raise ValueError(f"context sentences must be a list: {title}")
        sentences = [str(sentence) for sentence in raw_sentences]
        previous = content_by_title.get(title)
        if previous is not None:
            if previous != sentences:
                raise ValueError(f"conflicting context paragraphs share title: {title}")
            continue
        content_by_title[title] = sentences
        deduplicated.append((title, sentences))
    return deduplicated


def load_musique(
    path: str | Path,
    *,
    limit: int,
    seed: int,
) -> list[BenchmarkExample]:
    raw = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows, _ = _sample_rows(raw, limit=limit, seed=seed)
    return [
        BenchmarkExample(
            row["id"],
            row["question"],
            row["answer"],
            benchmark="musique",
            context=tuple(
                BenchmarkContext(
                    str(paragraph["idx"]),
                    f"{paragraph['title']}\n{paragraph['paragraph_text']}",
                )
                for paragraph in row["paragraphs"]
            ),
            aliases=tuple(row["answer_aliases"]),
            supporting_source_ids=frozenset(
                str(paragraph["idx"])
                for paragraph in row["paragraphs"]
                if paragraph["is_supporting"]
            ),
            metadata={
                "answerable": str(row["answerable"]),
                "hops": str(len(row["question_decomposition"])),
            },
        )
        for row in rows
    ]


def _sample_parquet(
    path: str | Path,
    *,
    limit: int,
    seed: int,
    columns: list[str] | None = None,
) -> tuple[list[dict], list[int]]:
    table = _parquet_table(path, columns=columns)
    indices = _sample_indices(table.num_rows, limit=limit, seed=seed)
    return table.take(indices).to_pylist(), indices


def _read_parquet(
    path: str | Path,
    *,
    columns: list[str] | None = None,
) -> tuple[list[dict], list[int]]:
    table = _parquet_table(path, columns=columns)
    indices = list(range(table.num_rows))
    return table.to_pylist(), indices


def _parquet_table(
    path: str | Path,
    *,
    columns: list[str] | None,
):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "benchmark parquet loading requires the 'bench' dependencies"
        ) from exc
    return pq.read_table(Path(path), columns=columns)


def _sample_rows(
    rows: list[RowT],
    *,
    limit: int,
    seed: int,
) -> tuple[list[RowT], list[int]]:
    indices = _sample_indices(len(rows), limit=limit, seed=seed)
    return [rows[index] for index in indices], indices


def _sample_indices(size: int, *, limit: int, seed: int) -> list[int]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > size:
        raise ValueError(f"limit {limit} exceeds dataset size {size}")
    return sorted(random.Random(seed).sample(range(size), limit))
