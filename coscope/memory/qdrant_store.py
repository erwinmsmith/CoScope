"""Qdrant-backed memory store with scope filters applied inside vector search."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from coscope.config import MemorySettings
from coscope.core.artifact import ArtifactState
from coscope.core.memory import MemoryEntry
from coscope.memory.store import MemoryStoreStats, SearchHit
from coscope.scope.descriptor import Lifecycle, ScopeDescriptor, Visibility
from coscope.scope.effective_view import EffectiveView
from coscope.scope.permissions import Permission

_CLIENTS: dict[tuple[str, str, float], QdrantClient] = {}
_CLIENTS_LOCK = threading.Lock()
_INITIALIZED_COLLECTIONS: set[tuple[str, str, int]] = set()
_COLLECTIONS_LOCK = threading.Lock()
_NEVER_EXPIRES = 253_402_300_799.0


def build_qdrant_store(
    settings: MemorySettings,
    *,
    dimension: int,
    namespace: str | None = None,
) -> QdrantMemoryStore:
    """Build a task-isolated store over a shared Qdrant collection."""
    client = _shared_client(settings)
    return QdrantMemoryStore(
        client,
        collection_name=settings.qdrant_collection,
        dimension=dimension,
        namespace=namespace,
        initialize_key=settings.qdrant_url,
    )


def _shared_client(settings: MemorySettings) -> QdrantClient:
    key = (
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.timeout_seconds,
    )
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            if settings.qdrant_url == ":memory:":
                client = QdrantClient(location=":memory:")
            else:
                client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    timeout=max(1, int(settings.timeout_seconds)),
                )
            _CLIENTS[key] = client
        return client


class QdrantMemoryStore:
    """Persist every memory write and read through Qdrant."""

    vector_name = "dense"

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str,
        dimension: int,
        namespace: str | None = None,
        initialize_key: str = "client",
    ):
        if dimension <= 0:
            raise ValueError("Qdrant vector dimension must be positive")
        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension
        self.namespace = namespace or f"store_{uuid.uuid4().hex}"
        self._revision = 0
        self._stats = MemoryStoreStats()
        self._ensure_collection(initialize_key)
        # A deterministic namespace can be reused after an interrupted task.
        # Clear only that task's points before starting the new attempt.
        self.clear()

    def _ensure_collection(self, initialize_key: str) -> None:
        key = (initialize_key, self.collection_name, self.dimension)
        with _COLLECTIONS_LOCK:
            if key in _INITIALIZED_COLLECTIONS:
                return
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    self.collection_name,
                    vectors_config={
                        self.vector_name: models.VectorParams(
                            size=self.dimension,
                            distance=models.Distance.COSINE,
                        )
                    },
                )
            else:
                collection = self.client.get_collection(self.collection_name)
                vectors = collection.config.params.vectors
                config = (
                    vectors.get(self.vector_name)
                    if isinstance(vectors, dict)
                    else vectors
                )
                actual = int(config.size) if config is not None else 0
                if actual != self.dimension:
                    raise RuntimeError(
                        "Qdrant collection dimension mismatch: "
                        f"expected {self.dimension}, found {actual}"
                    )
            if initialize_key != ":memory:":
                for field, schema in (
                    ("store_id", models.PayloadSchemaType.KEYWORD),
                    ("scope_id", models.PayloadSchemaType.KEYWORD),
                    ("state", models.PayloadSchemaType.KEYWORD),
                    ("memory_types", models.PayloadSchemaType.KEYWORD),
                    ("tenant_id", models.PayloadSchemaType.KEYWORD),
                    ("workspace_id", models.PayloadSchemaType.KEYWORD),
                    ("searchable", models.PayloadSchemaType.BOOL),
                    ("valid_until_sort", models.PayloadSchemaType.FLOAT),
                ):
                    self.client.create_payload_index(
                        self.collection_name,
                        field,
                        field_schema=schema,
                        wait=True,
                    )
            _INITIALIZED_COLLECTIONS.add(key)

    def add(self, entry: MemoryEntry) -> None:
        if self.get(entry.memory_id) is not None:
            raise ValueError(f"duplicate memory: {entry.memory_id}")
        self._upsert(entry)

    def replace(self, entry: MemoryEntry) -> None:
        self._upsert(entry)

    def _upsert(self, entry: MemoryEntry) -> None:
        searchable = entry.vector is not None
        vector = (
            list(entry.vector)
            if entry.vector is not None
            else [0.0] * self.dimension
        )
        if len(vector) != self.dimension:
            raise ValueError(
                f"memory vector dimension must be {self.dimension}, "
                f"received {len(vector)}"
            )
        self.client.upsert(
            self.collection_name,
            [
                models.PointStruct(
                    id=self._point_id(entry.memory_id),
                    vector={self.vector_name: vector},
                    payload=self._payload(entry, searchable=searchable),
                )
            ],
            wait=True,
        )
        self._revision += 1

    def get(self, memory_id: str) -> MemoryEntry | None:
        records = self.client.retrieve(
            self.collection_name,
            [self._point_id(memory_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return None
        record = records[0]
        payload = dict(record.payload or {})
        if payload.get("store_id") != self.namespace:
            return None
        return self._entry(record)

    def scopes(self) -> list[ScopeDescriptor]:
        scopes: dict[str, ScopeDescriptor] = {}
        for entry in self:
            scopes[entry.scope.scope_id] = entry.scope
        return list(scopes.values())

    def entries_in_view(
        self,
        view: EffectiveView,
        *,
        states: frozenset[ArtifactState] | None = None,
        memory_types: frozenset[str] | None = None,
    ) -> list[MemoryEntry]:
        if not view.scope_ids:
            return []
        allowed_states = states or frozenset(
            {ArtifactState.VERIFIED, ArtifactState.COMMITTED}
        )
        return list(
            self._scroll(
                self._view_filter(
                    view,
                    states=allowed_states,
                    memory_types=memory_types,
                )
            )
        )

    def search(
        self,
        vector: tuple[float, ...],
        view: EffectiveView,
        *,
        top_k: int,
        memory_types: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0 or not view.scope_ids:
            return []
        if len(vector) != self.dimension:
            raise ValueError("vector dimensions differ")
        query_filter = self._view_filter(
            view,
            states=frozenset(
                {ArtifactState.VERIFIED, ArtifactState.COMMITTED}
            ),
            memory_types=memory_types,
            searchable=True,
        )
        started = time.perf_counter()
        response = self.client.query_points(
            self.collection_name,
            query=list(vector),
            using=self.vector_name,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        result = [
            SearchHit(self._entry(point), float(point.score))
            for point in response.points
        ]
        self._stats.search_calls += 1
        self._stats.search_seconds += time.perf_counter() - started
        return result

    def clear(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            self.collection_name,
            models.FilterSelector(filter=self._store_filter()),
            wait=True,
        )
        self._revision += 1

    def __len__(self) -> int:
        return int(
            self.client.count(
                self.collection_name,
                count_filter=self._store_filter(),
                exact=True,
            ).count
        )

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def stats(self) -> MemoryStoreStats:
        return self._stats

    def __iter__(self) -> Iterator[MemoryEntry]:
        return self._scroll(self._store_filter())

    def _scroll(self, query_filter: models.Filter) -> Iterator[MemoryEntry]:
        offset: Any = None
        while True:
            records, offset = self.client.scroll(
                self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for record in records:
                yield self._entry(record)
            if offset is None:
                return

    def _store_filter(self) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="store_id",
                    match=models.MatchValue(value=self.namespace),
                )
            ]
        )

    def _view_filter(
        self,
        view: EffectiveView,
        *,
        states: frozenset[ArtifactState],
        memory_types: frozenset[str] | None,
        searchable: bool | None = None,
    ) -> models.Filter:
        must: list[Any] = [
            models.FieldCondition(
                key="store_id",
                match=models.MatchValue(value=self.namespace),
            ),
            models.FieldCondition(
                key="scope_id",
                match=models.MatchAny(any=sorted(view.scope_ids)),
            ),
            models.FieldCondition(
                key="state",
                match=models.MatchAny(
                    any=sorted(state.value for state in states)
                ),
            ),
            models.FieldCondition(
                key="valid_until_sort",
                range=models.Range(gt=time.time()),
            ),
        ]
        if memory_types:
            must.append(
                models.FieldCondition(
                    key="memory_types",
                    match=models.MatchAny(any=sorted(memory_types)),
                )
            )
        if searchable is not None:
            must.append(
                models.FieldCondition(
                    key="searchable",
                    match=models.MatchValue(value=searchable),
                )
            )
        return models.Filter(must=must)

    def _payload(
        self,
        entry: MemoryEntry,
        *,
        searchable: bool,
    ) -> dict[str, Any]:
        scope = entry.scope
        return {
            "store_id": self.namespace,
            "memory_id": entry.memory_id,
            "content": entry.content,
            "source_type": entry.source_type,
            "source_id": entry.source_id,
            "state": entry.state.value,
            "created_at": entry.created_at,
            "valid_until": entry.valid_until,
            "valid_until_sort": (
                entry.valid_until
                if entry.valid_until is not None
                else _NEVER_EXPIRES
            ),
            "searchable": searchable,
            "metadata": _json_value(entry.metadata),
            "scope_id": scope.scope_id,
            "tenant_id": scope.tenant_id,
            "workspace_id": scope.workspace_id,
            "memory_types": sorted(scope.memory_types),
            "scope": {
                "knowledge_domains": sorted(scope.knowledge_domains),
                "runtime_region": scope.runtime_region,
                "visibility": scope.visibility.value,
                "owner_id": scope.owner_id,
                "lifecycle": scope.lifecycle.value,
                "permissions": {
                    principal: sorted(permission.value for permission in values)
                    for principal, values in scope.permissions.items()
                },
                "memory_types": sorted(scope.memory_types),
                "tenant_id": scope.tenant_id,
                "workspace_id": scope.workspace_id,
                "policy_version": scope.policy_version,
                "index_snapshot": scope.index_snapshot,
                "provenance": dict(scope.provenance),
                "trust_level": scope.trust_level,
            },
        }

    def _entry(self, point: Any) -> MemoryEntry:
        payload = dict(point.payload or {})
        scope_raw = dict(payload["scope"])
        scope = ScopeDescriptor(
            knowledge_domains=frozenset(scope_raw["knowledge_domains"]),
            runtime_region=str(scope_raw["runtime_region"]),
            visibility=Visibility(str(scope_raw["visibility"])),
            owner_id=(
                str(scope_raw["owner_id"])
                if scope_raw.get("owner_id") is not None
                else None
            ),
            lifecycle=Lifecycle(str(scope_raw["lifecycle"])),
            permissions={
                str(principal): frozenset(
                    Permission(str(permission)) for permission in values
                )
                for principal, values in dict(
                    scope_raw.get("permissions", {})
                ).items()
            },
            memory_types=frozenset(scope_raw["memory_types"]),
            tenant_id=str(scope_raw["tenant_id"]),
            workspace_id=str(scope_raw["workspace_id"]),
            policy_version=str(scope_raw["policy_version"]),
            index_snapshot=str(scope_raw["index_snapshot"]),
            provenance={
                str(key): str(value)
                for key, value in dict(
                    scope_raw.get("provenance", {})
                ).items()
            },
            trust_level=str(scope_raw["trust_level"]),
        )
        vectors = point.vector
        dense = (
            vectors.get(self.vector_name)
            if isinstance(vectors, Mapping)
            else vectors
        )
        vector = (
            tuple(float(value) for value in dense)
            if payload.get("searchable") and dense is not None
            else None
        )
        return MemoryEntry(
            content=str(payload["content"]),
            scope=scope,
            source_type=str(payload["source_type"]),
            source_id=str(payload["source_id"]),
            state=ArtifactState(str(payload["state"])),
            memory_id=str(payload["memory_id"]),
            created_at=float(payload["created_at"]),
            valid_until=(
                float(payload["valid_until"])
                if payload.get("valid_until") is not None
                else None
            ),
            vector=vector,
            metadata=dict(payload.get("metadata", {})),
        )

    def _point_id(self, memory_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.namespace}:{memory_id}"))


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_value(item) for item in value]
    return str(value)
