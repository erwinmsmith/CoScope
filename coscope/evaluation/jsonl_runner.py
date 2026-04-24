"""
JSONL-driven Mode-B evaluation runner.

Design notes
------------
This is the "algorithm evaluation" path described in §14.5.1 of the
experimental design: we replay the ``retrieval_requests`` recorded in each
episode JSONL, run the CoScope retrieval variants (A1/A3/A4/A5/A8/...) on the
exact same memory snapshot, and compare the returned candidates against the
``ground_truth`` stored alongside. No LLM calls happen here; the only variable
is the retrieval pipeline itself, so the numbers are directly comparable
across variants and reproducible across machines (as long as the embedding
provider is deterministic).

Inputs
------
* One or more JSONL shard paths produced by ``coscope/scripts/build_all.py``.

Outputs
-------
* ``EpisodeRun``: per-episode breakdown, one entry per (episode, variant).
* ``StratifiedReport``: aggregated metrics grouped by (variant, rho_subset).
  S4 is filled from episodes with ``policy_conflict = True`` regardless of the
  stored ``rho_subset`` enum (see §14.2.4).

Boundaries (intentional)
------------------------
* We do not mutate the engine between episodes; a fresh ``CoScope`` is
  instantiated per episode so scopes / memories do not leak across tasks.
* Intermediate matrices from the 7-stage pipeline are kept accessible via the
  pipeline stats (``first_stage_actual``, ``shareable_buckets``, ...), the
  same fields the synthetic runner uses.
* Gold evidence is inverted from ``GroundTruthEvidence.required_by_agent_ids``
  onto ``request_id`` so that ``evaluate_retrieval`` can score each request.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from coscope.core.types import (
    AgentRole,
    Episode,
    MemoryEntry,
    MemoryType,
    RetrievalRequest,
    SubsetLabel,
)
from coscope.engine import CoScope
from coscope.evaluation.metrics import EvaluationReport
from coscope.evaluation.runner import (
    DEFAULT_VARIANTS,
    VariantRun,
    evaluate_variants,
)
from coscope.utils.output.serializer import Serializer as JSONLSerializer


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRun:
    """Per-episode evaluation result across all requested variants."""

    episode_id: str
    rho: float
    rho_subset: str
    policy_conflict: bool
    s4_eligible: bool
    graph_type: str
    n_requests: int
    n_memory_entries: int
    n_ground_truth: int
    variant_runs: Dict[str, VariantRun] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "rho": self.rho,
            "rho_subset": self.rho_subset,
            "policy_conflict": self.policy_conflict,
            "s4_eligible": self.s4_eligible,
            "graph_type": self.graph_type,
            "n_requests": self.n_requests,
            "n_memory_entries": self.n_memory_entries,
            "n_ground_truth": self.n_ground_truth,
            "variants": {v: run.to_row() for v, run in self.variant_runs.items()},
        }


@dataclass
class StratifiedCell:
    """Metrics aggregated over episodes sharing (variant, subset)."""

    variant: str
    subset: str
    n_episodes: int
    recall_at_k: float
    mrr_at_k: float
    first_stage_savings: float
    false_merge_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "subset": self.subset,
            "n_episodes": self.n_episodes,
            "recall_at_k": self.recall_at_k,
            "mrr_at_k": self.mrr_at_k,
            "first_stage_savings": self.first_stage_savings,
            "false_merge_rate": self.false_merge_rate,
        }


@dataclass
class StratifiedReport:
    """Full stratified report: (variant, subset) -> StratifiedCell."""

    k: int
    n_episodes_total: int
    cells: Dict[Tuple[str, str], StratifiedCell] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "n_episodes_total": self.n_episodes_total,
            "cells": [cell.to_dict() for cell in self.cells.values()],
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_episodes(shard_paths: Iterable[Path]) -> List[Episode]:
    """Read all episodes from the given JSONL shards (order preserved)."""
    serializer = JSONLSerializer()
    episodes: List[Episode] = []
    for path in shard_paths:
        episodes.extend(serializer.read_jsonl(Path(path)))
    return episodes


# ---------------------------------------------------------------------------
# Per-episode prep
# ---------------------------------------------------------------------------


def _build_gold_by_request(episode: Episode) -> Dict[str, List[str]]:
    """
    Invert GroundTruthEvidence.required_by_agent_ids onto request_id.

    Ground truth in the JSONL is stored as ``{memory_id, required_by_agent_ids}``.
    Evaluation metrics need ``{request_id -> [gold memory_ids]}``, so we join on
    ``agent_id``. Requests whose agent has no gold evidence are kept out of the
    gold map (they'll be skipped by recall / MRR in the standard way).
    """
    agent_to_request: Dict[str, str] = {
        r.agent_id: r.request_id for r in episode.retrieval_requests
    }
    gold: Dict[str, List[str]] = defaultdict(list)
    for evidence in episode.ground_truth:
        for agent_id in evidence.required_by_agent_ids:
            request_id = agent_to_request.get(agent_id)
            if request_id is None:
                continue
            gold[request_id].append(evidence.memory_id)
    return dict(gold)


def _conflict_request_ids(episode: Episode) -> List[str]:
    """
    Identify S4 / verifier requests that must never be merged into shared
    buckets. For episodes without policy conflict this is the empty list and
    FMR will be reported as 0 / no-op by evaluate_retrieval.
    """
    if not episode.policy_conflict:
        return []
    return [
        r.request_id
        for r in episode.retrieval_requests
        if r.role == AgentRole.VERIFIER
    ]


def _preload_memories(coscope: CoScope, episode: Episode) -> int:
    """
    Inject memory entries verbatim into the engine's memory store, preserving
    original ``memory_id`` values so that ground_truth references resolve. We
    bypass ``add_memory`` (which mints new IDs) and call ``memory_store.add``
    directly; embeddings are filled in with the engine's configured provider
    when missing.
    """
    added = 0
    for entry in episode.memory_entries:
        if entry.embedding is None:
            entry.embedding = coscope.embedding_provider.embed_query(entry.content)
        coscope.memory_store.add(entry)
        added += 1
    return added


def _register_episode_agents(coscope: CoScope, episode: Episode) -> None:
    """Register all agents recorded in the episode into the engine registry."""
    for agent in episode.agents:
        coscope.register_agent(agent.config)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _assign_subset(episode: Episode) -> str:
    """
    Resolve the evaluation subset for an episode. Policy-conflict episodes are
    reported under S4 regardless of their rho-based label (§14.2.4); all other
    episodes use their ``rho_subset`` enum.
    """
    if episode.policy_conflict:
        return SubsetLabel.S4.value
    if isinstance(episode.rho_subset, SubsetLabel):
        return episode.rho_subset.value
    return str(episode.rho_subset)


def _aggregate(
    runs: Sequence[Tuple[str, List[VariantRun]]],
    k: int,
) -> StratifiedReport:
    """
    Aggregate per-episode variant runs into (variant, subset) cells using a
    simple mean-of-means. Each episode contributes equally to its bucket,
    independent of request count, to avoid skewing by graph_type.
    """
    buckets: Dict[Tuple[str, str], List[EvaluationReport]] = defaultdict(list)
    for subset, variant_runs in runs:
        for vr in variant_runs:
            buckets[(vr.variant, subset)].append(vr.report)

    report = StratifiedReport(k=k, n_episodes_total=len(runs))
    for (variant, subset), reports in buckets.items():
        if not reports:
            continue
        n = len(reports)
        report.cells[(variant, subset)] = StratifiedCell(
            variant=variant,
            subset=subset,
            n_episodes=n,
            recall_at_k=sum(r.recall_at_k for r in reports) / n,
            mrr_at_k=sum(r.mrr_at_k for r in reports) / n,
            first_stage_savings=sum(r.first_stage_savings for r in reports) / n,
            false_merge_rate=sum(r.false_merge_rate for r in reports) / n,
        )
    return report


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_jsonl(
    shard_paths: Sequence[Path],
    *,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    k: int = 10,
    engine_factory: Optional[callable] = None,
    verbose: bool = False,
) -> Tuple[List[EpisodeRun], StratifiedReport]:
    """
    Run Mode-B evaluation over the given JSONL shards.

    Parameters
    ----------
    shard_paths:
        JSONL files produced by the construction pipeline. Each line is a
        complete episode.
    variants:
        Pipeline variants to evaluate (see ``DEFAULT_VARIANTS``).
    k:
        Cutoff for Recall@k / MRR@k.
    engine_factory:
        Callable returning a fresh ``CoScope`` engine; defaults to
        ``CoScope()``. Override to swap embedding provider (e.g. sentence-
        transformers or dashscope text-embedding-v3).
    verbose:
        If True, emit a one-line progress message per episode.

    Returns
    -------
    ``(episode_runs, stratified_report)`` where ``episode_runs`` preserves
    per-episode detail and ``stratified_report`` is the (variant, subset)
    aggregate ready for §14.5.2 tables.
    """
    if engine_factory is None:
        engine_factory = CoScope  # default engine, random (hash) embeddings

    episodes = load_episodes([Path(p) for p in shard_paths])

    episode_runs: List[EpisodeRun] = []
    per_subset_runs: List[Tuple[str, List[VariantRun]]] = []

    for idx, episode in enumerate(episodes):
        gold = _build_gold_by_request(episode)
        conflict_ids = _conflict_request_ids(episode)

        if not episode.retrieval_requests or not gold:
            if verbose:
                print(
                    f"[skip] {episode.episode_id}: "
                    f"n_requests={len(episode.retrieval_requests)} "
                    f"n_gold={len(gold)}"
                )
            continue

        coscope = engine_factory()
        _register_episode_agents(coscope, episode)
        _preload_memories(coscope, episode)

        variant_runs = evaluate_variants(
            coscope=coscope,
            requests=episode.retrieval_requests,
            gold_by_request=gold,
            variants=list(variants),
            k=k,
            conflict_request_ids=conflict_ids if conflict_ids else None,
            independent_first_stage=len(episode.retrieval_requests),
        )

        subset = _assign_subset(episode)
        episode_runs.append(
            EpisodeRun(
                episode_id=episode.episode_id,
                rho=episode.rho,
                rho_subset=subset,
                policy_conflict=episode.policy_conflict,
                s4_eligible=episode.s4_eligible,
                graph_type=(
                    episode.graph_type.value
                    if hasattr(episode.graph_type, "value")
                    else str(episode.graph_type)
                ),
                n_requests=len(episode.retrieval_requests),
                n_memory_entries=len(episode.memory_entries),
                n_ground_truth=len(episode.ground_truth),
                variant_runs={vr.variant: vr for vr in variant_runs},
            )
        )
        per_subset_runs.append((subset, list(variant_runs)))

        if verbose:
            recalls = ", ".join(
                f"{vr.variant.upper()}={vr.report.recall_at_k:.3f}"
                for vr in variant_runs
            )
            print(
                f"[{idx + 1:04d}] {episode.episode_id} "
                f"subset={subset} rho={episode.rho:.3f} "
                f"reqs={len(episode.retrieval_requests)} "
                f"gold={sum(len(v) for v in gold.values())} "
                f"{recalls}"
            )

    stratified = _aggregate(per_subset_runs, k=k)
    return episode_runs, stratified


# ---------------------------------------------------------------------------
# Pretty printing (matches §14.5.2 table shape)
# ---------------------------------------------------------------------------


def format_stratified_table(
    report: StratifiedReport,
    *,
    metric: str = "recall_at_k",
    digits: int = 4,
    subsets: Sequence[str] = ("S1", "S2", "S3", "S4"),
) -> str:
    """Render a ``variant x subset`` Markdown table for the given metric."""
    variants = sorted({variant for (variant, _) in report.cells.keys()})
    if not variants:
        return "(empty report)"

    header = ["Variant"] + list(subsets) + ["N"]
    rows: List[List[str]] = []
    for variant in variants:
        row = [variant.upper()]
        total_n = 0
        for subset in subsets:
            cell = report.cells.get((variant, subset))
            if cell is None:
                row.append("-")
                continue
            total_n += cell.n_episodes
            value = getattr(cell, metric)
            row.append(f"{value:.{digits}f}")
        row.append(str(total_n))
        rows.append(row)

    widths = [
        max(len(str(header[i])), *(len(str(row[i])) for row in rows))
        for i in range(len(header))
    ]
    header_line = "| " + " | ".join(
        str(h).ljust(widths[i]) for i, h in enumerate(header)
    ) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(header))) + " |"
    row_lines = [
        "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([f"Metric: {metric}  (k={report.k})", header_line, sep_line, *row_lines])


__all__ = [
    "EpisodeRun",
    "StratifiedCell",
    "StratifiedReport",
    "evaluate_jsonl",
    "format_stratified_table",
    "load_episodes",
]
