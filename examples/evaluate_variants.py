"""
Run A1/A3/A4/A5 retrieval variants and print a compact comparison table.
"""

from __future__ import annotations

import os

from engine import CoScope
from core.types import MemoryType, PolicyConstraints, VisibilityLevel
from evaluation import evaluate_variants, format_variant_table


def build_toy_case():
    """Create a small case with shared evidence and one policy-conflict request."""
    coscope = CoScope()

    team_policy = PolicyConstraints(
        visibility=[VisibilityLevel.TEAM],
        max_clearance=1,
    )
    restricted_policy = PolicyConstraints(
        visibility=[VisibilityLevel.RESTRICTED],
        max_clearance=5,
        audit_required=True,
    )

    coscope.create_agent(
        agent_id="planner_1",
        role="planner",
        allowed_scopes=["task/shared"],
        allowed_memory_types=[MemoryType.SEMANTIC],
        policy=team_policy,
    )
    coscope.create_agent(
        agent_id="solver_1",
        role="solver",
        allowed_scopes=["task/shared"],
        allowed_memory_types=[MemoryType.SEMANTIC],
        policy=team_policy,
    )
    coscope.create_agent(
        agent_id="verifier_1",
        role="verifier",
        allowed_scopes=["task/shared"],
        allowed_memory_types=[MemoryType.SEMANTIC],
        policy=restricted_policy,
    )

    gold = coscope.add_memory(
        content="The alpha task constraint is backed by approved API evidence.",
        scope_id="task/shared",
        memory_type=MemoryType.SEMANTIC,
        confidence=0.95,
        visibility=[VisibilityLevel.TEAM],
    )
    coscope.add_memory(
        content="A beta note that is unrelated to the alpha constraint.",
        scope_id="task/shared",
        memory_type=MemoryType.SEMANTIC,
        confidence=0.35,
        visibility=[VisibilityLevel.TEAM],
    )

    requests = [
        coscope.create_request(
            "planner_1",
            "What is the alpha task constraint?",
        ),
        coscope.create_request(
            "solver_1",
            "Which evidence supports the alpha constraint?",
        ),
        coscope.create_request(
            "verifier_1",
            "Audit restricted provenance for the alpha evidence.",
        ),
    ]

    gold_by_request = {
        requests[0].request_id: [gold.memory_id],
        requests[1].request_id: [gold.memory_id],
    }
    conflict_request_ids = [requests[2].request_id]

    return coscope, requests, gold_by_request, conflict_request_ids


def main() -> None:
    os.environ.setdefault("COSCOPE_LOG_LEVEL", "WARNING")
    coscope, requests, gold_by_request, conflict_request_ids = build_toy_case()
    runs = evaluate_variants(
        coscope,
        requests,
        gold_by_request,
        k=2,
        conflict_request_ids=conflict_request_ids,
    )
    print(format_variant_table(runs))


if __name__ == "__main__":
    main()
