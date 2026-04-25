"""
Basic Usage Example for CoScope.

Demonstrates fundamental usage of CoScope for collaborative
memory retrieval in a multi-agent system.
"""

import sys
sys.path.insert(0, "/Users/erwin/Downloads/codespace/CoScope")

from engine import CoScope
from core.types import AgentRole, AgentState, MemoryType, ScopeSpec, VisibilityLevel
from core.scope import PolicyConstraints
from core.types import Provenance


def main():
    # ============================================================
    # Initialize CoScope Engine
    # ============================================================
    print("Initializing CoScope engine...")

    coscope = CoScope()

    # ============================================================
    # Register Agents
    # ============================================================
    print("\nRegistering agents...")

    planner = coscope.create_agent(
        agent_id="planner_1",
        role="planner",
        name="Task Planner",
        description="Responsible for task planning and constraint analysis",
        allowed_scopes=["task/shared", "session/current", "workspace/project_x"],
    )

    solver = coscope.create_agent(
        agent_id="solver_1",
        role="solver",
        name="Task Solver",
        description="Responsible for solving sub-tasks",
        allowed_scopes=["task/shared", "session/current"],
    )

    verifier = coscope.create_agent(
        agent_id="verifier_1",
        role="verifier",
        name="Evidence Verifier",
        description="Responsible for verifying evidence",
        allowed_scopes=["task/shared", "session/current", "governed/audit"],
    )

    print(f"Registered {len(coscope.list_agents())} agents")

    # ============================================================
    # Add Memory Entries
    # ============================================================
    print("\nAdding memory entries...")

    coscope.add_memory(
        content="The main constraint is to complete the task within 2 hours and use only approved APIs.",
        scope_id="task/shared",
        memory_type=MemoryType.SEMANTIC,
        visibility=[VisibilityLevel.TEAM],
        tags=["constraint", "planning"],
    )

    coscope.add_memory(
        content="The solution approach will use a divide-and-conquer strategy.",
        scope_id="task/shared",
        memory_type=MemoryType.SEMANTIC,
        visibility=[VisibilityLevel.TEAM],
        tags=["strategy", "planning"],
    )

    coscope.add_memory(
        content="Evidence: The API documentation confirms that the endpoint supports batch requests.",
        scope_id="task/shared",
        memory_type=MemoryType.EPISODIC,
        visibility=[VisibilityLevel.TEAM],
        provenance=Provenance(
            source="api_documentation",
            agent_id="verifier_1",
        ),
        tags=["evidence", "api"],
    )

    coscope.add_memory(
        content="Current progress: Phase 1 completed, Phase 2 in progress.",
        scope_id="session/current",
        memory_type=MemoryType.EPISODIC,
        visibility=[VisibilityLevel.TEAM],
        tags=["progress", "status"],
    )

    print("Added memories to store")

    # ============================================================
    # Create Retrieval Requests
    # ============================================================
    print("\nCreating retrieval requests...")

    requests = [
        coscope.create_request(
            agent_id="planner_1",
            query="What are the key constraints and dependencies for the current task?",
            priority=1,
        ),
        coscope.create_request(
            agent_id="solver_1",
            query="What factual evidence supports the solution approach?",
            priority=1,
        ),
        coscope.create_request(
            agent_id="verifier_1",
            query="What is the provenance of the evidence we're using?",
            priority=2,
        ),
    ]

    print(f"Created {len(requests)} retrieval requests")

    # ============================================================
    # Execute Collaborative Retrieval
    # ============================================================
    print("\nExecuting collaborative retrieval...")

    results = coscope.retrieve(requests, fit_projection=True)

    # ============================================================
    # Display Results
    # ============================================================
    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    for result in results:
        print(f"\n--- Agent: {result.agent_id} ({result.role.value}) ---")
        print(f"Fallback triggered: {result.fallback_triggered}")
        print(f"Candidates: {len(result.candidates)}")

        for i, candidate in enumerate(result.candidates[:3], 1):
            print(f"  {i}. [{candidate.memory.memory_type.value}] Score: {candidate.score:.3f}")
            print(f"     Content: {candidate.memory.content[:60]}...")

    # ============================================================
    # Get Engine Statistics
    # ============================================================
    print("\n" + "=" * 60)
    print("ENGINE STATISTICS")
    print("=" * 60)

    stats = coscope.get_stats()
    print(f"Number of agents: {stats['num_agents']}")
    print(f"Number of memories: {stats['num_memories']}")


if __name__ == "__main__":
    main()
