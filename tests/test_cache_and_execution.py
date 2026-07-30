import pytest

from coscope import AgentInstance, CoScopeRuntime, MemoryEntry
from coscope.core import Artifact, ArtifactState, ArtifactType
from coscope.reasoning import ReasoningConfig, ReasoningMode
from coscope.replay import MockExecutor
from coscope.retrieval import RetrievalCandidate, SharedRetrievalCache
from coscope.runtime import RuntimeInvocation
from coscope.scope import ScopeDescriptor, Visibility


def _configured_runtime() -> tuple[CoScopeRuntime, str, ScopeDescriptor, ScopeDescriptor]:
    runtime = CoScopeRuntime()
    runtime.create_run("task", run_id="run")
    runtime.register_agent(
        AgentInstance(
            "solver",
            "solver",
            knowledge_permissions=frozenset({"docs"}),
        )
    )
    _, node = runtime.start_reasoning(
        "solver", ReasoningConfig(ReasoningMode.COT)
    )
    public = ScopeDescriptor(
        frozenset({"docs"}), "run/task", Visibility.TEAM_SHARED
    )
    private = ScopeDescriptor(
        frozenset({"docs"}),
        node.runtime_region,
        Visibility.AGENT_PRIVATE,
        owner_id="solver",
    )
    runtime.ingest_memory(
        MemoryEntry("public deployment evidence", public, "document", "doc")
    )
    # Register the private scope before the invocation writes an artifact.
    runtime.ingest_memory(
        MemoryEntry("existing private note", private, "agent", "old")
    )
    return runtime, node.node_id, public, private


def test_shared_cache_reuses_only_identical_scope_bound_queries():
    runtime, node_id, _, _ = _configured_runtime()
    first = runtime.create_request(
        run_id="run",
        agent_id="solver",
        reasoning_node_id=node_id,
        full_query="deployment evidence",
        public_intent="deployment evidence",
    )
    runtime.retrieve_batch([first])
    assert runtime.retrieval.stats.shared_store_queries == 1

    second = runtime.create_request(
        run_id="run",
        agent_id="solver",
        reasoning_node_id=node_id,
        full_query="deployment evidence",
        public_intent="deployment evidence",
    )
    runtime.retrieve_batch([second])
    assert runtime.retrieval.stats.shared_cache_hits == 1
    assert runtime.retrieval.stats.shared_store_queries == 0


def test_memory_mutation_invalidates_scope_bound_cache():
    runtime, node_id, public, _ = _configured_runtime()
    first = runtime.create_request(
        run_id="run",
        agent_id="solver",
        reasoning_node_id=node_id,
        full_query="deployment evidence",
        public_intent="deployment evidence",
    )
    runtime.retrieve_batch([first])
    runtime.ingest_memory(
        MemoryEntry("new deployment evidence", public, "document", "new_doc")
    )
    second = runtime.create_request(
        run_id="run",
        agent_id="solver",
        reasoning_node_id=node_id,
        full_query="deployment evidence",
        public_intent="deployment evidence",
    )
    runtime.retrieve_batch([second])
    assert runtime.retrieval.stats.shared_cache_hits == 0
    assert runtime.retrieval.stats.shared_store_queries == 1


def test_shared_cache_rejects_private_candidates():
    runtime, _, _, private = _configured_runtime()
    memory = next(entry for entry in runtime.memory if entry.scope == private)
    cache = SharedRetrievalCache()
    with pytest.raises(ValueError, match="private"):
        cache.put(
            object(),  # type: ignore[arg-type]
            [RetrievalCandidate(memory, 1.0, "private")],
        )


def test_agent_rerank_vector_is_reused_across_final_merge():
    runtime, node_id, _, _ = _configured_runtime()
    request = runtime.create_request(
        run_id="run",
        agent_id="solver",
        reasoning_node_id=node_id,
        full_query="deployment evidence",
        public_intent="deployment evidence",
    )
    result = runtime.retrieve_batch([request])[request.request_id]
    assert request.rerank_vector is not None
    first_vector = request.rerank_vector
    runtime.retrieval.reranker.rerank(request, result.candidates)
    assert request.rerank_vector is first_vector


def test_simulated_executor_writes_proposed_private_artifact():
    runtime, node_id, _, private = _configured_runtime()

    def handler(agent, node, context):
        return [
            Artifact(
                "new simulated finding",
                ArtifactType.FACT,
                "mock",
                node.node_id,
                agent.agent_id,
                state=ArtifactState.PROPOSED,
            )
        ]

    outputs = runtime.execute_batch(
        "run",
        [
            RuntimeInvocation(
                "solver",
                node_id,
                "deployment evidence",
                "deployment evidence",
                output_scope=private,
            )
        ],
        MockExecutor(handler),
    )
    assert len(outputs) == 1
    assert any(
        entry.content == "new simulated finding"
        and entry.state == ArtifactState.PROPOSED
        for entry in runtime.memory
    )
