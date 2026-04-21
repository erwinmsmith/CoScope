"""Solver agent builder (one per SOLVER node in the graph)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from coscope.core.types import (
    Agent,
    AgentRole,
    PolicyConstraints,
    RetrievalRequest,
    VisibilityLevel,
)
from coscope.data.agents.common import (
    SOLVER_MEMORY_TYPES,
    _make_agent,
    _make_request,
)
from coscope.data.core.types import GoTGraph, GoTNode


class SolverAgentBuilder:
    """Produces Solver Agents + retrieval requests for every SOLVER node."""

    def build_all(
        self,
        raw_item: Dict[str, Any],
        got_graph: GoTGraph,
        episode_id: str,
        dataset: str,
    ) -> List[Tuple[Agent, RetrievalRequest]]:
        results: List[Tuple[Agent, RetrievalRequest]] = []
        for node in got_graph.solver_nodes():
            results.append(self.build(raw_item, got_graph, node, episode_id, dataset))
        return results

    def build(
        self,
        raw_item: Dict[str, Any],
        got_graph: GoTGraph,
        node: GoTNode,
        episode_id: str,
        dataset: str,
    ) -> Tuple[Agent, RetrievalRequest]:
        policy = PolicyConstraints(
            visibility=[VisibilityLevel.TEAM, VisibilityLevel.PUBLIC],
            max_clearance=1,
            excluded_zones=["quarantine", "audit_hold"],
            audit_required=False,
        )
        agent = _make_agent(
            episode_id=episode_id,
            dataset=dataset,
            node=node,
            got_graph=got_graph,
            role=AgentRole.SOLVER,
            memory_types=SOLVER_MEMORY_TYPES,
            policy=policy,
            include_restricted=False,
            extra_metadata={"agent_kind": "solver"},
        )
        query = self._build_query(raw_item, node)
        request = _make_request(
            agent,
            dataset=dataset,
            episode_id=episode_id,
            node_id=node.node_id,
            role=AgentRole.SOLVER,
            hop_index=node.hop_index,
            query=query,
            memory_types=SOLVER_MEMORY_TYPES,
            include_restricted=False,
        )
        return agent, request

    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(raw_item: Dict[str, Any], node: GoTNode) -> str:
        """
        Solver-k query generation:
            - QA with sub_questions: use sub_questions[k-1] (fall back to question)
            - QA without sub_questions (HotpotQA / 2Wiki when missing): reuse the
              main question so the Solver at least has anchor text.
            - Math: first sentence of solution_steps[k-1] (without numerics section).
        """
        hop_index = node.hop_index or 1
        sub_qs: List[str] = raw_item.get("sub_questions", []) or []
        solution_steps: List[str] = raw_item.get("solution_steps", []) or []

        if raw_item.get("dataset_type") == "math" and solution_steps:
            idx = min(hop_index - 1, len(solution_steps) - 1)
            step = solution_steps[idx]
            head = step.split(".", 1)[0].strip()
            return head or step
        if sub_qs:
            idx = min(hop_index - 1, len(sub_qs) - 1)
            return sub_qs[idx]
        return raw_item.get("question", "") or ""
