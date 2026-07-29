"""Reasoning-state node."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from coscope.reasoning.modes import ReasoningMode


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    PRUNED = "pruned"
    FAILED = "failed"


@dataclass
class ReasoningNode:
    node_id: str
    reasoning_mode: ReasoningMode
    owner_agent_id: str
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    runtime_region: str = ""
    status: NodeStatus = NodeStatus.PENDING
    local_state: dict[str, object] = field(default_factory=dict)
    context_requirements: dict[str, object] = field(default_factory=dict)
