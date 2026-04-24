"""Shared helpers for building agents from GoT nodes."""

from __future__ import annotations

from typing import Dict, List, Optional

from coscope.core.types import (
    Agent,
    AgentConfig,
    AgentRole,
    AgentState,
    MemoryType,
    PolicyConstraints,
    RetrievalRequest,
    ScopeSpec,
    VisibilityLevel,
)
from coscope.core.types import GoTGraph, GoTNode
from coscope.core.scope_ids import (
    agent_private,
    task_restricted,
    task_shared,
    workspace_semantic,
)


SOLVER_MEMORY_TYPES: List[MemoryType] = [
    MemoryType.EPISODIC,
    MemoryType.SEMANTIC,
    MemoryType.ARTIFACT,
]
PLANNER_MEMORY_TYPES: List[MemoryType] = [MemoryType.EPISODIC, MemoryType.SEMANTIC]
VERIFIER_MEMORY_TYPES: List[MemoryType] = [MemoryType.EPISODIC, MemoryType.SEMANTIC]


def _workspace_scope_for_role(
    dataset: str, role: AgentRole, hop_index: Optional[int]
) -> str:
    """v3 (flat): every role sees the single flat corpus scope."""
    return workspace_semantic(dataset)


def _scope_spec(
    dataset: str,
    episode_id: str,
    node_id: str,
    role: AgentRole,
    hop_index: Optional[int],
    *,
    include_restricted: bool = False,
) -> ScopeSpec:
    """Produce the ScopeSpec used for an agent's retrieval request."""
    return ScopeSpec(
        private_scopes=[agent_private(episode_id, node_id)],
        shared_scopes=[task_shared(episode_id)],
        workspace_scopes=[_workspace_scope_for_role(dataset, role, hop_index)],
        governed_scopes=[task_restricted(episode_id)] if include_restricted else [],
    )


def _allowed_scopes(
    dataset: str,
    episode_id: str,
    node_id: str,
    role: AgentRole,
    hop_index: Optional[int],
    *,
    include_restricted: bool = False,
) -> List[str]:
    scopes = [
        _workspace_scope_for_role(dataset, role, hop_index),
        task_shared(episode_id),
        agent_private(episode_id, node_id),
    ]
    if include_restricted:
        scopes.append(task_restricted(episode_id))
    return scopes


def _make_agent(
    *,
    episode_id: str,
    dataset: str,
    node: GoTNode,
    got_graph: GoTGraph,
    role: AgentRole,
    memory_types: List[MemoryType],
    policy: PolicyConstraints,
    include_restricted: bool,
    extra_metadata: Dict[str, object],
) -> Agent:
    agent_id = f"{episode_id}_{node.node_id}"
    allowed_scopes = _allowed_scopes(
        dataset,
        episode_id,
        node.node_id,
        role,
        node.hop_index,
        include_restricted=include_restricted,
    )
    metadata: Dict[str, object] = {
        "node_id": node.node_id,
        "hop_index": node.hop_index,
        "episode_id": episode_id,
        "dataset": dataset,
        "accessible_ancestor_node_ids": sorted(
            set(got_graph.get_all_ancestors(node.node_id)) | {node.node_id}
        ),
    }
    metadata.update(extra_metadata)

    config = AgentConfig(
        agent_id=agent_id,
        role=role,
        name=node.node_id,
        description=f"{role.value} agent for node {node.node_id}",
        allowed_scopes=allowed_scopes,
        allowed_memory_types=list(memory_types),
        policy=policy,
        metadata=metadata,
    )
    initial_state = AgentState(
        plan_node=node.node_id,
        task_context={"episode_id": episode_id, "dataset": dataset},
    )
    return Agent(config=config, state=initial_state)


def _make_request(
    agent: Agent,
    *,
    dataset: str,
    episode_id: str,
    node_id: str,
    role: AgentRole,
    hop_index: Optional[int],
    query: str,
    memory_types: List[MemoryType],
    include_restricted: bool,
) -> RetrievalRequest:
    scope = _scope_spec(
        dataset,
        episode_id,
        node_id,
        role,
        hop_index,
        include_restricted=include_restricted,
    )
    return agent.create_retrieval_request(
        query=query,
        scope=scope,
        memory_types=list(memory_types),
    )
