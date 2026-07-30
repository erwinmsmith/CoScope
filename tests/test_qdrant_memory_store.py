from __future__ import annotations

import uuid

from qdrant_client import QdrantClient

from coscope.core import ArtifactState, MemoryEntry
from coscope.memory import QdrantMemoryStore
from coscope.scope import Permission, ScopeDescriptor, Visibility
from coscope.scope.effective_view import EffectiveView


def _store(
    client: QdrantClient,
    collection: str,
    namespace: str,
) -> QdrantMemoryStore:
    return QdrantMemoryStore(
        client,
        collection_name=collection,
        dimension=8,
        namespace=namespace,
        initialize_key=":memory:",
    )


def _view(*scopes: ScopeDescriptor) -> EffectiveView:
    first = scopes[0]
    public = frozenset(
        scope.scope_id
        for scope in scopes
        if scope.is_public_execution_scope
    )
    private = frozenset(scope.scope_id for scope in scopes) - public
    return EffectiveView(
        scope_ids=frozenset(scope.scope_id for scope in scopes),
        public_scope_ids=public,
        private_scope_ids=private,
        tenant_id=first.tenant_id,
        workspace_id=first.workspace_id,
        policy_versions=frozenset(scope.policy_version for scope in scopes),
        index_snapshots=frozenset(scope.index_snapshot for scope in scopes),
    )


def test_qdrant_store_round_trips_memory_without_an_in_memory_shadow() -> None:
    client = QdrantClient(location=":memory:")
    collection = f"memory_{uuid.uuid4().hex}"
    store = _store(client, collection, "run-a")
    scope = ScopeDescriptor(
        frozenset({"task"}),
        "run-a",
        Visibility.TEAM_SHARED,
        permissions={
            "planner": frozenset({Permission.READ, Permission.WRITE})
        },
        memory_types=frozenset({"benchmark_evidence"}),
    )
    entry = MemoryEntry(
        "evidence",
        scope,
        "benchmark_context",
        "source-a",
        vector=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metadata={"roles": ["planner"]},
    )

    store.add(entry)
    loaded = store.get(entry.memory_id)

    assert loaded == entry
    assert len(store) == 1
    assert next(iter(store)) == entry
    assert store.scopes() == [scope]
    hits = store.search(
        entry.vector or (),
        _view(scope),
        top_k=5,
    )
    assert [hit.memory.memory_id for hit in hits] == [entry.memory_id]
    assert store.stats.search_calls == 1
    assert store.stats.search_seconds > 0

    loaded.state = ArtifactState.VERIFIED
    store.replace(loaded)
    assert store.get(entry.memory_id).state == ArtifactState.VERIFIED


def test_qdrant_search_filters_scope_and_tenant_before_vector_ranking() -> None:
    client = QdrantClient(location=":memory:")
    collection = f"memory_{uuid.uuid4().hex}"
    store = _store(client, collection, "run-filter")
    allowed = ScopeDescriptor(
        frozenset({"public"}),
        "run-filter",
        Visibility.TEAM_SHARED,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )
    forbidden = ScopeDescriptor(
        frozenset({"private"}),
        "run-filter",
        Visibility.RESTRICTED,
        permissions={"solver": frozenset({Permission.READ})},
        tenant_id="tenant-b",
        workspace_id="workspace-b",
    )
    vector = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    store.add(
        MemoryEntry(
            "allowed",
            allowed,
            "evidence",
            "allowed",
            vector=vector,
        )
    )
    store.add(
        MemoryEntry(
            "forbidden",
            forbidden,
            "evidence",
            "forbidden",
            vector=vector,
        )
    )

    hits = store.search(vector, _view(allowed), top_k=10)

    assert [hit.memory.content for hit in hits] == ["allowed"]
    assert [
        entry.content for entry in store.entries_in_view(_view(allowed))
    ] == ["allowed"]


def test_qdrant_keeps_nonsearchable_output_readable_but_not_retrievable() -> None:
    client = QdrantClient(location=":memory:")
    collection = f"memory_{uuid.uuid4().hex}"
    store = _store(client, collection, "run-output")
    scope = ScopeDescriptor(
        frozenset({"thinking"}),
        "run-output",
        Visibility.TEAM_SHARED,
    )
    output = MemoryEntry(
        "published summary",
        scope,
        "published_thinking",
        "summary",
        vector=None,
    )
    store.add(output)

    assert store.get(output.memory_id).content == "published summary"
    assert [
        entry.content for entry in store.entries_in_view(_view(scope))
    ] == ["published summary"]
    assert store.search((1.0,) + (0.0,) * 7, _view(scope), top_k=5) == []


def test_qdrant_namespace_cleanup_cannot_delete_another_run() -> None:
    client = QdrantClient(location=":memory:")
    collection = f"memory_{uuid.uuid4().hex}"
    first = _store(client, collection, "run-first")
    second = _store(client, collection, "run-second")
    scope = ScopeDescriptor(
        frozenset({"task"}),
        "shared",
        Visibility.TEAM_SHARED,
    )
    vector = (1.0,) + (0.0,) * 7
    first.add(MemoryEntry("first", scope, "evidence", "first", vector=vector))
    second.add(MemoryEntry("second", scope, "evidence", "second", vector=vector))

    first.clear()

    assert len(first) == 0
    assert len(second) == 1
    assert next(iter(second)).content == "second"
