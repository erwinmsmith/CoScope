import pytest

from coscope import AgentInstance, CoScopeRuntime, MemoryEntry, ReasoningConfig, ReasoningMode
from coscope.core import Artifact, ArtifactState, ArtifactType
from coscope.scope import Permission, ScopeDescriptor, Visibility


def _runtime_fixture():
    runtime = CoScopeRuntime()
    runtime.create_run("task", run_id="run")
    planner = AgentInstance(
        "planner", "planner", knowledge_permissions=frozenset({"docs"})
    )
    verifier = AgentInstance(
        "verifier", "verifier", knowledge_permissions=frozenset({"docs"})
    )
    runtime.register_agent(planner)
    runtime.register_agent(verifier)
    _, planner_node = runtime.start_reasoning(
        "planner", ReasoningConfig(ReasoningMode.COT)
    )
    _, verifier_node = runtime.start_reasoning(
        "verifier", ReasoningConfig(ReasoningMode.COT)
    )
    shared = ScopeDescriptor(
        frozenset({"docs"}), "run/task", Visibility.TEAM_SHARED
    )
    private = ScopeDescriptor(
        frozenset({"docs"}),
        planner_node.runtime_region,
        Visibility.AGENT_PRIVATE,
        owner_id="planner",
    )
    restricted = ScopeDescriptor(
        frozenset({"docs"}),
        verifier_node.runtime_region,
        Visibility.RESTRICTED,
        permissions={"role:verifier": frozenset({Permission.READ})},
    )
    runtime.ingest_memory(
        MemoryEntry("public project deadline schedule", shared, "corpus", "public")
    )
    runtime.ingest_memory(
        MemoryEntry("private contract deadline", private, "agent", "private")
    )
    runtime.ingest_memory(
        MemoryEntry("restricted audit finding", restricted, "audit", "restricted")
    )
    return runtime, planner_node, verifier_node, shared, private, restricted


def test_public_reuse_and_private_fallback_do_not_expand_access():
    runtime, planner_node, verifier_node, _, _, restricted = _runtime_fixture()
    planner_request = runtime.create_request(
        run_id="run",
        agent_id="planner",
        reasoning_node_id=planner_node.node_id,
        full_query="project deadline and private contract",
        public_intent="project deadline",
        private_intent="private contract",
    )
    verifier_request = runtime.create_request(
        run_id="run",
        agent_id="verifier",
        reasoning_node_id=verifier_node.node_id,
        full_query="project deadline and audit",
        public_intent="project deadline",
        private_intent="audit finding",
    )
    results = runtime.retrieve_batch([planner_request, verifier_request])

    assert runtime.retrieval.stats.shared_store_queries == 1
    assert results[planner_request.request_id].group_id == results[verifier_request.request_id].group_id
    planner_ids = {
        item.memory.scope.scope_id
        for item in results[planner_request.request_id].candidates
    }
    verifier_ids = {
        item.memory.scope.scope_id
        for item in results[verifier_request.request_id].candidates
    }
    assert restricted.scope_id not in planner_ids
    assert restricted.scope_id in verifier_ids


def test_public_intent_must_be_explicit_and_cannot_embed_declared_private_text():
    runtime, planner_node, _, _, _, _ = _runtime_fixture()
    with pytest.raises(ValueError, match="public_intent"):
        runtime.create_request(
            run_id="run",
            agent_id="planner",
            reasoning_node_id=planner_node.node_id,
            full_query="contains secret",
            private_intent="secret",
        )
    with pytest.raises(ValueError, match="contains"):
        runtime.create_request(
            run_id="run",
            agent_id="planner",
            reasoning_node_id=planner_node.node_id,
            full_query="contains secret",
            public_intent="contains secret",
            private_intent="secret",
        )


def test_promotion_requires_verified_state_and_explicit_permission():
    runtime, planner_node, _verifier_node, _, private, _ = _runtime_fixture()
    shared_target = ScopeDescriptor(
        frozenset({"docs"}),
        "run/task/committed",
        Visibility.TEAM_SHARED,
        permissions={"role:verifier": frozenset({Permission.PROMOTE})},
    )
    artifact = Artifact(
        "checked conclusion",
        ArtifactType.FACT,
        "agent",
        planner_node.node_id,
        "planner",
        state=ArtifactState.PROPOSED,
    )
    entry = runtime.write_artifact(
        "planner", artifact, private, runtime_region=planner_node.runtime_region
    )
    with pytest.raises(ValueError, match="verified"):
        runtime.promote_memory("verifier", entry.memory_id, shared_target)
    runtime.verify_memory(entry.memory_id)
    promoted = runtime.promote_memory("verifier", entry.memory_id, shared_target)
    assert promoted.state == ArtifactState.COMMITTED
    assert promoted.scope.scope_id == shared_target.scope_id


def test_traces_do_not_store_query_or_memory_content():
    runtime, planner_node, _, _, _, _ = _runtime_fixture()
    request = runtime.create_request(
        run_id="run",
        agent_id="planner",
        reasoning_node_id=planner_node.node_id,
        full_query="project deadline",
        public_intent="project deadline",
    )
    runtime.retrieve_batch([request])
    attributes = [event.attributes for event in runtime.traces.events("run")]
    assert all("full_query" not in item and "content" not in item for item in attributes)
