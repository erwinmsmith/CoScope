"""
CoScope Data: GoT-MAS dataset construction subpackage.

Public API:
    build_episode(raw_item, dataset, split, target_graph_type, seed) -> Episode | None
    load_episodes(dataset, split, subsets=None, graph_types=None) -> List[Episode]
    DatasetPipeline: batch construction over a full dataset split.

See `data/docs/GoT_Dataset_Requirements_v2.md` for the full specification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from coscope.data.core.types import (
    Episode,
    GoTEdge,
    GoTGraph,
    GoTNode,
    GraphType,
    GroundTruthEvidence,
    InvalidGraphError,
    ReasoningPathType,
    SubsetLabel,
)
from coscope.data.output.serializer import Serializer
from coscope.data.output.stats_reporter import (
    DatasetQualityError,
    StatsReporter,
    SubsetCoverageError,
)
from coscope.data.pipeline.dataset_pipeline import DatasetPipeline
from coscope.data.pipeline.episode_builder import EpisodeBuilder


def build_episode(
    raw_item: Dict[str, Any],
    dataset: str,
    split: str,
    target_graph_type: Union[str, GraphType],
    seed: int = 42,
) -> Optional[Episode]:
    """Build a single Episode. Returns None if validation fails."""
    return EpisodeBuilder().build_episode(
        raw_item=raw_item,
        dataset=dataset,
        split=split,
        target_graph_type=target_graph_type,
        seed=seed,
    )


def load_episodes(
    dataset: str,
    split: str,
    *,
    subsets: Optional[Sequence[str]] = None,
    graph_types: Optional[Sequence[str]] = None,
    processed_dir: Union[str, Path] = "data/processed",
) -> List[Episode]:
    """
    Load previously serialized episodes from `data/processed/<dataset>/<split>/`.

    Args:
        dataset: dataset name (e.g. "musique").
        split:   "train" | "dev" | "test".
        subsets: optional filter, e.g. ["S1", "S2", "S4"].
        graph_types: optional filter on graph_type values (e.g. ["LINEAR"]).
        processed_dir: override for the processed root directory.
    """
    root = Path(processed_dir) / dataset / split
    if not root.exists():
        return []
    serializer = Serializer()
    wanted_subsets = {s.lower() for s in subsets} if subsets else None
    wanted_graph_types = {g.lower() for g in graph_types} if graph_types else None

    out: List[Episode] = []
    for file in sorted(root.glob("*.jsonl")):
        stem = file.stem   # e.g. "s1_linear"
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        subset_tag, graph_tag = parts[0], parts[1]
        if wanted_subsets and subset_tag not in wanted_subsets:
            continue
        if wanted_graph_types and graph_tag not in wanted_graph_types:
            continue
        out.extend(serializer.read_jsonl(file))
    return out


__all__ = [
    # Public API
    "build_episode",
    "load_episodes",
    "DatasetPipeline",
    "EpisodeBuilder",
    "Serializer",
    "StatsReporter",
    "SubsetCoverageError",
    "DatasetQualityError",
    # Re-exported types
    "Episode",
    "GoTGraph",
    "GoTNode",
    "GoTEdge",
    "GraphType",
    "GroundTruthEvidence",
    "InvalidGraphError",
    "ReasoningPathType",
    "SubsetLabel",
]
