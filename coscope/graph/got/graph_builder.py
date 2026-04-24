"""
GoT graph builder.

Thin orchestrator over `graph_templates.build_graph`. Accepts a raw_item dict
and picks the right hop_count from it, then invokes the structural template.
"""

from __future__ import annotations

from typing import Any, Dict, Union

from coscope.core.types import GoTGraph, GraphType, InvalidGraphError
from coscope.graph.got.graph_templates import build_graph


class GraphBuilder:
    """Construct a GoTGraph for a raw_item under a target graph_type."""

    def __init__(self, seed: int = 42):
        self.seed = seed  # kept for API compatibility; deterministic templates don't need it

    def build(
        self,
        raw_item: Dict[str, Any],
        dataset: str,
        target_graph_type: Union[str, GraphType],
        seed: int = 42,
    ) -> GoTGraph:
        if isinstance(target_graph_type, str):
            try:
                target_graph_type = GraphType(target_graph_type)
            except ValueError as exc:
                raise InvalidGraphError(
                    f"Unknown graph_type: {target_graph_type!r}"
                ) from exc
        hop_count = int(raw_item.get("hop_count", 2))
        try:
            return build_graph(dataset=dataset, hop_count=hop_count, graph_type=target_graph_type)
        except InvalidGraphError:
            raise
        except Exception as exc:
            raise InvalidGraphError(
                f"Failed to build graph for dataset={dataset!r} "
                f"hop_count={hop_count} graph_type={target_graph_type}: {exc}"
            ) from exc
