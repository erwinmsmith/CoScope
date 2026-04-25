"""
CoT chain builder.

Per `docs/GoT_Dataset_Requirements_v2.md` §18.4, CoT is a strict linear
chain `Planner -> Solver_1 -> ... -> Solver_k`, which is structurally the
degenerate LINEAR case of the GoT graph family. Rather than duplicating the
graph construction logic, this builder is a thin wrapper over
`graph.got.graph_builder.GraphBuilder` pinned to `GraphType.LINEAR`.

Keeping the wrapper (instead of directly calling `GraphBuilder`) preserves
extensibility: if CoT evolves to need extra metadata (e.g. step-level
pre-conditions), the entry point is already separated out.
"""

from __future__ import annotations

from typing import Any, Dict

from core.types import GoTGraph, GraphType
from graph.got.graph_builder import GraphBuilder


class ChainBuilder:
    """Construct a linear CoT reasoning chain for a raw_item.

    The returned graph is a `GoTGraph` with `graph_type=LINEAR`. Downstream
    episode construction uses the `reasoning_path_type=COT` tag on the episode
    to distinguish CoT episodes from GoT/LINEAR episodes.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._delegate = GraphBuilder(seed=seed)

    def build(self, raw_item: Dict[str, Any], dataset: str, seed: int = 42) -> GoTGraph:
        """Build a linear chain (LINEAR GoT template) for this raw_item."""
        return self._delegate.build(
            raw_item=raw_item,
            dataset=dataset,
            target_graph_type=GraphType.LINEAR,
            seed=seed,
        )


__all__ = ["ChainBuilder"]
