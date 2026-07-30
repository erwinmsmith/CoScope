"""Stable signature for shared-retrieval compatibility."""

from __future__ import annotations

import hashlib
import json

from coscope.scope.effective_view import EffectiveView


def scope_signature(
    view: EffectiveView,
    *,
    memory_types: frozenset[str],
    embedding_model_version: str,
    retrieval_parameters: dict[str, object] | None = None,
) -> str:
    payload = {
        "tenant": view.tenant_id,
        "workspace": view.workspace_id,
        "scopes": sorted(view.public_scope_ids),
        "memory_types": sorted(memory_types),
        "policy_versions": sorted(view.policy_versions),
        "index_snapshots": sorted(view.index_snapshots),
        "embedding_model": embedding_model_version,
        "retrieval": retrieval_parameters or {},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()
