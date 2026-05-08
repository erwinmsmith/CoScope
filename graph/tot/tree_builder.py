"""
ToT tree builder.

Per `docs/GoT_Dataset_Requirements_v2.md` §18.5, ToT is a tree-shaped
reasoning structure with fan-out but no merge; this corresponds structurally to
the GoT `FORK` graph template. This builder is a thin wrapper over
`graph.got.graph_builder.GraphBuilder` pinned to `GraphType.FORK`.

Downstream episodes are tagged with `reasoning_path_type=TOT` to distinguish
them from GoT/FORK episodes (same structure, different inference contract: ToT
adds an Evaluator step to pick the best branch, see `TOT_EVALUATOR_PROMPT`).

S4 (POLICY_ISOLATED) contract under ToT
---------------------------------------
``target_graph_type=POLICY_ISOLATED`` is a special case: instead of pinning
to FORK, we build a FORK base + Verifier node and tag the resulting graph as
``GraphType.POLICY_ISOLATED``. This produces a tree-shaped S4 episode that
shares the privacy contract of GoT POLICY_ISOLATED (Verifier holds restricted
``mem_rs_*`` evidence; solver agents must rely on private fallback) but with
a fan-out reasoning topology. The episode_id suffix becomes
``..._tot_POLICY_ISOLATED``, distinct from the GoT analogue
``..._got_POLICY_ISOLATED``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.types import GoTGraph, GraphType
from graph.got.graph_builder import GraphBuilder
from graph.got.graph_templates import (
    build_fork_pattern,
    build_policy_isolated,
)


class TreeBuilder:
    """Construct a branching ToT reasoning tree for a raw_item (GoT FORK template)."""

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
        """Build a fan-out tree for this raw_item.

        Two behaviours, dispatched on ``target_graph_type``:

        * ``POLICY_ISOLATED`` (S4): build a FORK base + Verifier node and
          tag the graph as POLICY_ISOLATED so subset_assigner /
          restricted_builder treat it as an S4 episode.
        * Any other value (or ``None``): build a plain FORK graph; the
          argument is otherwise ignored, mirroring the original wrapper
          contract.
        """
        if _is_policy_isolated(target_graph_type):
            hop_count = int(raw_item.get("hop_count", 2))
            return build_policy_isolated(build_fork_pattern, hop_count)
        return self._delegate.build(
            raw_item=raw_item,
            dataset=dataset,
            target_graph_type=GraphType.FORK,
            seed=seed,
        )


def _is_policy_isolated(value: Any) -> bool:
    if isinstance(value, GraphType):
        return value == GraphType.POLICY_ISOLATED
    return str(value or "").upper() == GraphType.POLICY_ISOLATED.value


__all__ = ["TreeBuilder"]
