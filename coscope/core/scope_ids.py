"""Canonical scope_id helper functions and layer constants."""

from __future__ import annotations


def workspace_semantic(dataset: str) -> str:
    """Flat workspace scope: accessible by all agents (v1, post-hop-removal)."""
    return f"workspace/{dataset}/semantic"


def task_shared(episode_id: str) -> str:
    return f"task/{episode_id}/shared"


def agent_private(episode_id: str, node_id: str) -> str:
    return f"task/{episode_id}/{node_id}/private"


def task_restricted(episode_id: str) -> str:
    return f"task/{episode_id}/restricted"


SCOPE_LAYERS = {
    "workspace_semantic",
    "task_shared_plan",
    "task_shared_artifact",
    "agent_private_intent",
    "agent_private",
    "restricted_audit",
    "workspace_semantic_global",
    "task_shared_episodic",
    "restricted",
}

LAYER_VISIBILITY = {
    "workspace_semantic":        "public",
    "workspace_semantic_global": "public",
    "task_shared_plan":          "team",
    "task_shared_artifact":      "team",
    "task_shared_episodic":      "team",
    "agent_private_intent":      "owner",
    "agent_private":             "owner",
    "restricted":                "restricted",
    "restricted_audit":          "restricted",
}
