"""
Advanced Usage Example for CoScope.

Demonstrates advanced features including:
- Custom routing strategies
- Multiple retrieval components
- Memory CRUD operations
- Prompt management
"""

import sys
sys.path.insert(0, "/Users/erwin/Downloads/codespace/CoScope")

from engine import CoScope
from core.types import AgentRole, MemoryType, VisibilityLevel
from memory.crud import MemoryCRUD, MemoryQuery
from prompts import PromptManager, RoleTemplates


def main():
    # ============================================================
    # Initialize CoScope
    # ============================================================
    print("Initializing CoScope...")
    coscope = CoScope()

    # Register agents
    coscope.create_agent(agent_id="planner_1", role="planner")
    coscope.create_agent(agent_id="solver_1", role="solver")
    coscope.create_agent(agent_id="verifier_1", role="verifier")

    print(f"Registered {len(coscoope.list_agents())} agents")

    # ============================================================
    # Using Prompt Management
    # ============================================================
    print("\n--- Prompt Management ---")

    prompt_manager = PromptManager()

    # Get role-specific template
    template = prompt_manager.get_template("role/planner")
    print(f"Planner template: {template.name}")

    # Render template
    rendered = prompt_manager.render(
        "role/planner",
        query="What constraints exist?",
        task_context="Building a distributed system",
    )
    print(f"Rendered: {rendered[:80]}...")

    # ============================================================
    # Using Memory CRUD
    # ============================================================
    print("\n--- Memory CRUD ---")

    crud = MemoryCRUD(
        memory_store=coscope.memory_store,
        embedding_provider=embedding_provider,
    )

    # Bulk create memories
    memories = crud.bulk_create([
        {
            "content": "Constraint: Task must complete in 2 hours",
            "scope_id": "task/shared",
            "memory_type": MemoryType.SEMANTIC,
            "tags": ["constraint"],
        },
        {
            "content": "Strategy: Use parallel processing",
            "scope_id": "task/shared",
            "memory_type": MemoryType.SEMANTIC,
            "tags": ["strategy"],
        },
        {
            "content": "Evidence: API supports batch mode",
            "scope_id": "task/shared",
            "memory_type": MemoryType.EPISODIC,
            "tags": ["evidence"],
        },
    ])

    print(f"Created {len(memories)} memories")

    # Query memories
    query = MemoryQuery(
        text="constraint",
        filter={
            "scope_ids": ["task/shared"],
        },
        limit=5,
    )

    results = crud.query(query)
    print(f"Query found {len(results)} memories")

    # Get stats
    stats = crud.get_stats()
    print(f"Memory stats: {stats}")

    # ============================================================
    # Custom Retrieval Pipeline
    # ============================================================
    print("\n--- Custom Pipeline ---")

    from retrieval.router import OverlapAwareRouter
    from retrieval.reranker import RoleAwareReranker
    from retrieval.fusion import ReciprocalRankFusion

    # Use overlap-aware routing
    router = OverlapAwareRouter(min_overlap_score=0.3)
    print(f"Router: {type(router).__name__}")

    # Use role-aware reranking
    reranker = RoleAwareReranker()
    print(f"Reranker: {type(reranker).__name__}")

    # Use RRF fusion
    fusion = ReciprocalRankFusion(k=60)
    print(f"Fusion: {type(fusion).__name__}")

    # ============================================================
    # Execute Retrieval
    # ============================================================
    print("\n--- Retrieval ---")

    requests = [
        coscope.create_request(agent_id="planner_1", query="What are the constraints?"),
        coscope.create_request(agent_id="solver_1", query="What evidence is available?"),
    ]

    results = coscope.retrieve(requests)

    for r in results:
        print(f"Agent: {r.agent_id}, Candidates: {len(r.candidates)}")

    print("\nAdvanced example complete!")


if __name__ == "__main__":
    main()
