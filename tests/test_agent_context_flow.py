from coscope import (
    AgentClass,
    AgentContextPolicy,
    CoScopeRuntime,
    MemoryEntry,
)
from coscope.core import ArtifactType
from coscope.reasoning import ReasoningConfig, ReasoningMode
from coscope.scope import Permission, ScopeDescriptor, Visibility


def _agent_class(
    class_id: str,
    role: str,
    specialist_domain: str,
) -> AgentClass:
    return AgentClass(
        class_id,
        role,
        frozenset({"common", specialist_domain, "runtime_thinking"}),
        AgentContextPolicy(
            readable_artifact_types=frozenset(
                {ArtifactType.REASONING_SUMMARY}
            ),
            publishable_artifact_types=frozenset(
                {ArtifactType.REASONING_SUMMARY}
            ),
            allow_public_thinking=True,
        ),
    )


def test_agent_classes_resolve_different_knowledge_and_dynamic_thinking_flow():
    runtime = CoScopeRuntime()
    runtime.create_run("task", run_id="run")
    planner_class = _agent_class("planner_class", "planner", "planning")
    solver_class = _agent_class("solver_class", "solver", "solving")
    runtime.register_agent(planner_class.instantiate("planner"))
    runtime.register_agent(solver_class.instantiate("solver"))
    _, planner_node = runtime.start_reasoning(
        "planner", ReasoningConfig(ReasoningMode.COT)
    )
    _, solver_node = runtime.start_reasoning(
        "solver", ReasoningConfig(ReasoningMode.COT)
    )

    common = ScopeDescriptor(
        frozenset({"common"}),
        "run/task",
        Visibility.TEAM_SHARED,
    )
    planning = ScopeDescriptor(
        frozenset({"planning"}),
        "run/knowledge/planner",
        Visibility.RESTRICTED,
        permissions={"role:planner": frozenset({Permission.READ})},
    )
    solving = ScopeDescriptor(
        frozenset({"solving"}),
        "run/knowledge/solver",
        Visibility.RESTRICTED,
        permissions={"role:solver": frozenset({Permission.READ})},
    )
    runtime.ingest_memory(
        MemoryEntry("shared question", common, "user_input", "question")
    )
    runtime.ingest_memory(
        MemoryEntry(
            "planner-only knowledge",
            planning,
            "knowledge",
            "planner_knowledge",
            metadata={"retrieval_only": True},
        )
    )
    runtime.ingest_memory(
        MemoryEntry(
            "solver-only knowledge",
            solving,
            "knowledge",
            "solver_knowledge",
            metadata={"retrieval_only": True},
        )
    )

    planner_request = runtime.create_request(
        run_id="run",
        agent_id="planner",
        reasoning_node_id=planner_node.node_id,
        full_query="solve with planning knowledge",
        public_intent="solve shared question",
        private_intent="planning knowledge",
    )
    solver_request = runtime.create_request(
        run_id="run",
        agent_id="solver",
        reasoning_node_id=solver_node.node_id,
        full_query="solve with calculation knowledge",
        public_intent="solve shared question",
        private_intent="calculation knowledge",
    )
    results = runtime.retrieve_batch([planner_request, solver_request])
    assert runtime.retrieval.stats.groups == 1

    planner_sources = {
        candidate.memory.source_id
        for candidate in results[planner_request.request_id].candidates
    }
    solver_sources = {
        candidate.memory.source_id
        for candidate in results[solver_request.request_id].candidates
    }
    assert "planner_knowledge" in planner_sources
    assert "solver_knowledge" not in planner_sources
    assert "solver_knowledge" in solver_sources
    assert "planner_knowledge" not in solver_sources

    private_entry, public_entry = runtime.record_thinking(
        actor_id="planner",
        reasoning_node_id=planner_node.node_id,
        private_content="raw planner working text",
        public_summary="planner share-safe summary",
        recipients=frozenset({"solver"}),
    )
    assert public_entry is not None
    assert private_entry.vector is None
    assert private_entry.metadata["searchable"] is False
    assert public_entry.vector is None
    assert public_entry.metadata["searchable"] is False

    runtime.refresh_context_view(solver_request)
    solver_packet = runtime.assemble_context(
        solver_request,
        results[solver_request.request_id],
    )
    solver_memory_ids = {entry.memory_id for entry in solver_packet.all_memories}
    assert public_entry.memory_id in solver_memory_ids
    assert private_entry.memory_id not in solver_memory_ids

    runtime.refresh_context_view(planner_request)
    planner_packet = runtime.assemble_context(
        planner_request,
        results[planner_request.request_id],
    )
    planner_memory_ids = {
        entry.memory_id for entry in planner_packet.all_memories
    }
    assert private_entry.memory_id in planner_memory_ids


def test_agent_class_can_forbid_public_thinking():
    runtime = CoScopeRuntime()
    runtime.create_run("task", run_id="run")
    private_only = AgentClass(
        "private_only",
        "worker",
        frozenset({"runtime_thinking"}),
    )
    runtime.register_agent(private_only.instantiate("worker"))
    _, node = runtime.start_reasoning(
        "worker", ReasoningConfig(ReasoningMode.COT)
    )

    try:
        runtime.record_thinking(
            actor_id="worker",
            reasoning_node_id=node.node_id,
            private_content="private",
            public_summary="must not publish",
            recipients=None,
        )
    except PermissionError as exc:
        assert "cannot publish thinking" in str(exc)
    else:
        raise AssertionError("public thinking should have been rejected")
