"""Canonical scope_id helpers used across all memory builders."""

from __future__ import annotations


def workspace_semantic(dataset: str) -> str:
    """Global workspace scope: accessible by Planner and Verifier."""
    return f"workspace/{dataset}/semantic"


def workspace_semantic_hop(dataset: str, hop_index: int) -> str:
    """Per-hop workspace scope: accessible by Solver-k only (k == hop_index)."""
    return f"workspace/{dataset}/semantic/hop_{int(hop_index)}"


def task_shared(episode_id: str) -> str:
    return f"task/{episode_id}/shared"


def agent_private(episode_id: str, node_id: str) -> str:
    return f"task/{episode_id}/{node_id}/private"


def task_restricted(episode_id: str) -> str:
    return f"task/{episode_id}/restricted"


SCOPE_LAYERS = {
    "workspace_semantic_global",
    "workspace_semantic_hop",
    "task_shared_episodic",
    "agent_private",
    "restricted",
}


LAYER_VISIBILITY = {
    "workspace_semantic_global": "public",
    "workspace_semantic_hop": "public",
    "task_shared_episodic": "team",
    "agent_private": "private",
    "restricted": "restricted",
}
