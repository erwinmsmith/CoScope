"""Multi-dimensional scope descriptors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from coscope.scope.permissions import Permission


class Visibility(str, Enum):
    SYSTEM = "system"
    TEAM_SHARED = "team_shared"
    ROLE_SHARED = "role_shared"
    AGENT_PRIVATE = "agent_private"
    BRANCH_PRIVATE = "branch_private"
    NODE_LOCAL = "node_local"
    RESTRICTED = "restricted"


class Lifecycle(str, Enum):
    EPHEMERAL = "ephemeral"
    STEP = "step"
    BRANCH = "branch"
    RUN = "run"
    SESSION = "session"
    PERSISTENT = "persistent"


@dataclass(frozen=True)
class ScopeDescriptor:
    knowledge_domains: frozenset[str]
    runtime_region: str
    visibility: Visibility
    owner_id: str | None = None
    lifecycle: Lifecycle = Lifecycle.RUN
    permissions: Mapping[str, frozenset[Permission]] = field(default_factory=dict)
    memory_types: frozenset[str] = field(default_factory=lambda: frozenset({"semantic"}))
    tenant_id: str = "default"
    workspace_id: str = "default"
    policy_version: str = "1"
    index_snapshot: str = "current"
    provenance: Mapping[str, str] = field(default_factory=dict)
    trust_level: str = "unknown"

    @property
    def scope_id(self) -> str:
        payload = {
            "domains": sorted(self.knowledge_domains),
            "region": self.runtime_region,
            "visibility": self.visibility.value,
            "owner": self.owner_id,
            "tenant": self.tenant_id,
            "workspace": self.workspace_id,
            "policy": self.policy_version,
            "snapshot": self.index_snapshot,
            "types": sorted(self.memory_types),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        return f"scope_{digest}"

    @property
    def is_public_execution_scope(self) -> bool:
        return self.visibility in {
            Visibility.SYSTEM,
            Visibility.TEAM_SHARED,
            Visibility.ROLE_SHARED,
        }
