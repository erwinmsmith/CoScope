"""Planner agent builder."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from coscope.core.types import (
    Agent,
    AgentRole,
    PolicyConstraints,
    RetrievalRequest,
    VisibilityLevel,
)
from coscope.data.agents.common import (
    PLANNER_MEMORY_TYPES,
    _make_agent,
    _make_request,
)
from coscope.data.core.types import GoTGraph


class PlannerAgentBuilder:
    """Produces a Planner Agent + its initial retrieval request."""

    def build(
        self,
        raw_item: Dict[str, Any],
        got_graph: GoTGraph,
        episode_id: str,
        dataset: str,
    ) -> Tuple[Agent, RetrievalRequest]:
        planner_node = got_graph.planner_node()
        if planner_node is None:
            raise ValueError(f"GoTGraph missing PLANNER node (episode={episode_id})")

        policy = PolicyConstraints(
            visibility=[VisibilityLevel.TEAM, VisibilityLevel.PUBLIC],
            max_clearance=1,
            excluded_zones=["quarantine", "audit_hold"],
            audit_required=False,
        )
        agent = _make_agent(
            episode_id=episode_id,
            dataset=dataset,
            node=planner_node,
            got_graph=got_graph,
            role=AgentRole.PLANNER,
            memory_types=PLANNER_MEMORY_TYPES,
            policy=policy,
            include_restricted=False,
            extra_metadata={"agent_kind": "planner"},
        )
        query = self._build_query(raw_item)
        request = _make_request(
            agent,
            dataset=dataset,
            episode_id=episode_id,
            node_id=planner_node.node_id,
            role=AgentRole.PLANNER,
            hop_index=planner_node.hop_index,
            query=query,
            memory_types=PLANNER_MEMORY_TYPES,
            include_restricted=False,
        )
        return agent, request

    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(raw_item: Dict[str, Any]) -> str:
        question = raw_item.get("question", "") or ""
        if raw_item.get("dataset_type") == "math":
            return f"Plan the solution steps for the problem: {question}"
        return question
