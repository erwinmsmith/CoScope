from coscope.context import ContextManager
from coscope.core import ArtifactState, MemoryEntry
from coscope.memory import RuntimeMemoryStore
from coscope.reasoning import (
    CoTRuntime,
    GoTRuntime,
    NodeStatus,
    ReasoningConfig,
    ReasoningMode,
    ToTRuntime,
)
from coscope.retrieval import RetrievalCandidate, RetrievalRequest, RetrievalResult
from coscope.scope import EffectiveView, ScopeDescriptor, Visibility


def test_agent_topology_is_not_reasoning_mode():
    cot = CoTRuntime("planner", ReasoningConfig(ReasoningMode.COT, max_depth=3))
    first = cot.create_root()
    second = cot.advance()
    assert first.child_ids == [second.node_id]
    assert all(node.owner_agent_id == "planner" for node in cot.nodes())

    tot = ToTRuntime("planner", ReasoningConfig(ReasoningMode.TOT, branching_factor=2))
    root = tot.create_root()
    branches = tot.branch(root.node_id)
    assert len(branches) == 2
    assert branches[0].runtime_region != branches[1].runtime_region


def test_got_node_waits_for_all_parents():
    got = GoTRuntime("verifier")
    got.add_node("a")
    got.add_node("b")
    merged = got.add_node("merge", parent_ids=["a", "b"])
    got.complete("a")
    assert merged.status == NodeStatus.PENDING
    got.complete("b")
    assert merged.status == NodeStatus.READY
    assert got.visible_ancestor_ids("merge") == {"a", "b"}


def test_context_deduplicates_and_rejects_quarantined_items():
    scope = ScopeDescriptor(
        frozenset({"docs"}), "run/task", Visibility.TEAM_SHARED
    )
    first = MemoryEntry("same evidence", scope, "test", "1", vector=(1.0, 0.0))
    duplicate = MemoryEntry("same evidence", scope, "test", "2", vector=(1.0, 0.0))
    quarantined = MemoryEntry(
        "unsafe",
        scope,
        "test",
        "3",
        state=ArtifactState.QUARANTINED,
        vector=(0.0, 1.0),
    )
    store = RuntimeMemoryStore()
    for entry in (first, duplicate, quarantined):
        store.add(entry)
    view = EffectiveView(
        frozenset({scope.scope_id}),
        frozenset({scope.scope_id}),
        frozenset(),
        "default",
        "default",
        frozenset({"1"}),
        frozenset({"current"}),
    )
    request = RetrievalRequest(
        "req",
        "run",
        "agent",
        "node",
        "query",
        "query",
        None,
        (1.0, 0.0),
        "signature",
        view,
    )
    result = RetrievalResult(
        "req",
        [
            RetrievalCandidate(first, 1.0, "shared"),
            RetrievalCandidate(duplicate, 0.9, "shared"),
            RetrievalCandidate(quarantined, 0.8, "shared"),
        ],
    )
    packet = ContextManager(store).assemble(request, result)
    assert [entry.content for entry in packet.all_memories] == ["same evidence"]
