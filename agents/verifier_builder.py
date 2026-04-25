"""Verifier agent builder (only present in POLICY_ISOLATED graphs)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from core.types import (
    Agent,
    AgentRole,
    PolicyConstraints,
    RetrievalRequest,
    VisibilityLevel,
)
from agents.common import (
    VERIFIER_MEMORY_TYPES,
    _make_agent,
    _make_request,
)
from core.types import GoTGraph


VERIFIER_QUERY_TEMPLATES = {
    "musique": "Verify the credibility of each hop's supporting evidence and temporal consistency of the reasoning chain.",
    "hotpotqa": "Verify the credibility of each hop's supporting evidence and temporal consistency of the reasoning chain.",
    "2wikimhqa_bridge": "Verify the credibility of each hop's supporting evidence and temporal consistency of the reasoning chain.",
    "2wikimhqa_comparison": "Verify that the comparison evidence for both entities is authoritative and non-conflicting.",
    "2wikimhqa_bridge-comparison": "Verify the logical consistency of the reasoning steps and completeness of the evidence.",
    "2wikimhqa_bridge_comparison": "Verify the logical consistency of the reasoning steps and completeness of the evidence.",
    "gsm8k": "Verify that each step's numeric result is consistent with the gold intermediate values.",
    "math_algebra": "Verify equation transformations across steps and numeric self-consistency.",
    "math_geometry": "Verify the correctness of geometric relations and property citations.",
    "math_number_theory": "Verify the application of divisibility, congruence and similar properties.",
    "math_counting_and_probability": "Verify sample-space completeness and probability calculation correctness.",
    "math_intermediate_algebra": "Verify algebraic transformations and equation-solving consistency.",
    "math_prealgebra": "Verify basic arithmetic and quantitative relationships.",
    "math_precalculus": "Verify applications of functions, limits and trigonometric identities.",
}


class VerifierAgentBuilder:
    """Produces Verifier Agent + retrieval request when a VERIFIER node exists."""

    def build(
        self,
        raw_item: Dict[str, Any],
        got_graph: GoTGraph,
        episode_id: str,
        dataset: str,
    ) -> Optional[Tuple[Agent, RetrievalRequest]]:
        verifier_node = got_graph.verifier_node()
        if verifier_node is None:
            return None

        policy = PolicyConstraints(
            visibility=[VisibilityLevel.TEAM, VisibilityLevel.RESTRICTED],
            max_clearance=2,
            excluded_zones=["quarantine", "audit_hold"],
            audit_required=True,
        )
        agent = _make_agent(
            episode_id=episode_id,
            dataset=dataset,
            node=verifier_node,
            got_graph=got_graph,
            role=AgentRole.VERIFIER,
            memory_types=VERIFIER_MEMORY_TYPES,
            policy=policy,
            include_restricted=True,
            extra_metadata={"agent_kind": "verifier"},
        )
        query = self._build_query(raw_item, dataset)
        request = _make_request(
            agent,
            dataset=dataset,
            episode_id=episode_id,
            node_id=verifier_node.node_id,
            role=AgentRole.VERIFIER,
            hop_index=verifier_node.hop_index,
            query=query,
            memory_types=VERIFIER_MEMORY_TYPES,
            include_restricted=True,
        )
        return agent, request

    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(raw_item: Dict[str, Any], dataset: str) -> str:
        qa_type = raw_item.get("qa_type")
        math_category = raw_item.get("math_category")
        if dataset == "2wikimhqa" and qa_type:
            key = f"2wikimhqa_{qa_type}"
            if key in VERIFIER_QUERY_TEMPLATES:
                return VERIFIER_QUERY_TEMPLATES[key]
        if dataset == "math" and math_category:
            key = f"math_{math_category}"
            if key in VERIFIER_QUERY_TEMPLATES:
                return VERIFIER_QUERY_TEMPLATES[key]
        return VERIFIER_QUERY_TEMPLATES.get(
            dataset, VERIFIER_QUERY_TEMPLATES["musique"]
        )
