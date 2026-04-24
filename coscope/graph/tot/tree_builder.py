"""
ToT tree builder.

Per `coscope/data/docs/GoT_Dataset_Requirements_v2.md` §18.5, ToT is a tree-shaped
reasoning structure with fan-out but no merge; this corresponds structurally to
the GoT `FORK` graph template. This builder is a thin wrapper over
`coscope.data.got.graph_builder.GraphBuilder` pinned to `GraphType.FORK`.

Downstream episodes are tagged with `reasoning_path_type=TOT` to distinguish
them from GoT/FORK episodes (same structure, different inference contract: ToT
adds an Evaluator step to pick the best branch, see `TOT_EVALUATOR_PROMPT`).
"""

from __future__ import annotations

from typing import Any, Dict, Union

from coscope.core.types import GoTGraph, GraphType, InvalidGraphError
from coscope.graph.got.graph_builder import GraphBuilder


class TreeBuilder:
    """Construct a branching ToT reasoning tree for a raw_item (GoT FORK template)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._delegate = GraphBuilder(seed=seed)

    def build(
        self,
        raw_item: Dict[str, Any],
        *,
        dataset: str,
        target_graph_type: Union[str, GraphType] = GraphType.FORK,
        seed: int = 42,
    ) -> GoTGraph:
        """Build a ToT graph.

        ToT is represented by the FORK template in the current codebase.
        POLICY_ISOLATED is also allowed so S4 can be constructed under the
        ToT reasoning-path tag.
        """
        if isinstance(target_graph_type, str):
            try:
                target_graph_type = GraphType(target_graph_type)
            except ValueError as exc:
                raise InvalidGraphError(
                    f"Unknown graph_type for ToT: {target_graph_type!r}"
                ) from exc
        if target_graph_type not in (GraphType.FORK, GraphType.POLICY_ISOLATED):
            raise InvalidGraphError(
                "ToT currently supports only FORK and POLICY_ISOLATED graph types"
            )
        return self._delegate.build(
            raw_item=raw_item,
            dataset=dataset,
            target_graph_type=target_graph_type,
            seed=seed,
        )


__all__ = ["TreeBuilder"]
