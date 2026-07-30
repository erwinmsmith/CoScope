"""Offline smoke test for the scope-aware runtime."""

from coscope import AgentInstance, CoScopeRuntime, MemoryEntry
from coscope.reasoning import ReasoningConfig, ReasoningMode
from coscope.scope import ScopeDescriptor, Visibility


def main() -> int:
    runtime = CoScopeRuntime()
    runtime.create_run("smoke", run_id="run_smoke")
    for agent_id, role in (("planner", "planner"), ("solver", "solver")):
        runtime.register_agent(
            AgentInstance(
                agent_id,
                role,
                knowledge_permissions=frozenset({"docs"}),
            )
        )
    _, planner_node = runtime.start_reasoning(
        "planner", ReasoningConfig(ReasoningMode.TOT)
    )
    _, solver_node = runtime.start_reasoning(
        "solver", ReasoningConfig(ReasoningMode.COT)
    )
    scope = ScopeDescriptor(
        frozenset({"docs"}), "run_smoke/task", Visibility.TEAM_SHARED
    )
    runtime.ingest_memory(
        MemoryEntry("The deployment deadline is Friday.", scope, "document", "doc_1")
    )
    requests = [
        runtime.create_request(
            run_id="run_smoke",
            agent_id="planner",
            reasoning_node_id=planner_node.node_id,
            full_query="Find the deployment deadline for planning.",
            public_intent="deployment deadline",
        ),
        runtime.create_request(
            run_id="run_smoke",
            agent_id="solver",
            reasoning_node_id=solver_node.node_id,
            full_query="Find the deployment deadline for execution.",
            public_intent="deployment deadline",
        ),
    ]
    results = runtime.retrieve_batch(requests)
    assert runtime.retrieval.stats.shared_store_queries == 1
    assert all(results[request.request_id].candidates for request in requests)
    assert all(
        runtime.assemble_context(request, results[request.request_id]).all_memories
        for request in requests
    )
    print("CoScope smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
