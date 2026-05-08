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

S4 (POLICY_ISOLATED) contract under CoT
---------------------------------------
``target_graph_type=POLICY_ISOLATED`` builds a LINEAR base + Verifier node
tagged as ``GraphType.POLICY_ISOLATED``. Structurally this matches the GoT
POLICY_ISOLATED graph (which is also LINEAR + Verifier), but the episode is
tagged with ``reasoning_path_type=COT`` so the resulting episode_id
(``..._cot_POLICY_ISOLATED``) is distinct from the GoT analogue and the two
runs can be reported side-by-side.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.types import GoTGraph, GraphType
from graph.got.graph_builder import GraphBuilder
from graph.got.graph_templates import (
    build_linear_pattern,
    build_policy_isolated,
)


class ChainBuilder:
    """Construct a linear CoT reasoning chain for a raw_item.

    The returned graph is a `GoTGraph` with `graph_type=LINEAR` for the
    default case. Downstream episode construction uses the
    `reasoning_path_type=COT` tag on the episode to distinguish CoT episodes
    from GoT/LINEAR episodes.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._delegate = GraphBuilder(seed=seed)

    def build(
        self,
        raw_item: Dict[str, Any],
        dataset: str,
        target_graph_type: Any = None,
        seed: int = 42,
    ) -> GoTGraph:
        """Build a linear chain for this raw_item.

        Two behaviours, dispatched on ``target_graph_type``:

        * ``POLICY_ISOLATED`` (S4): build a LINEAR base + Verifier node and
          tag the graph as POLICY_ISOLATED so subset_assigner /
          restricted_builder treat it as an S4 episode.
        * Any other value (or ``None``): build a plain LINEAR graph; the
          argument is otherwise ignored, mirroring the original wrapper
          contract.
        """
        if _is_policy_isolated(target_graph_type):
            hop_count = int(raw_item.get("hop_count", 2))
            return build_policy_isolated(build_linear_pattern, hop_count)
        return self._delegate.build(
            raw_item=raw_item,
            dataset=dataset,
            target_graph_type=GraphType.LINEAR,
            seed=seed,
        )


def _is_policy_isolated(value: Any) -> bool:
    if isinstance(value, GraphType):
        return value == GraphType.POLICY_ISOLATED
    return str(value or "").upper() == GraphType.POLICY_ISOLATED.value


__all__ = ["ChainBuilder"]
