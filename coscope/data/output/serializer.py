"""
Episode serializer.

Produces the on-disk JSON / JSONL representation described in spec §14.1, plus
a symmetric deserializer. Each Episode round-trips losslessly.

Design notes:
- `MemoryEntry.embedding` is intentionally dropped (set to null) during
  serialization. It is regenerated at runtime.
- `Agent` does not provide `to_dict`, so we serialize its config + state here.
- `RetrievalRequest` is serialized via the existing
  `coscope.core.types.RetrievalRequestModel`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from coscope.core.types import (
    Agent,
    AgentConfig,
    AgentRole,
    AgentState,
    MemoryEntry,
    MemoryType,
    PolicyConstraints,
    Provenance,
    RetrievalRequest,
    RetrievalRequestModel,
    ScopeSpec,
    VisibilityLevel,
    policy_constraints_from_dict,
)
from coscope.data.core.types import (
    Episode,
    GoTGraph,
    GraphType,
    GroundTruthEvidence,
    ReasoningPathType,
    SubsetLabel,
)


SCHEMA_VERSION = "2.0"


class Serializer:
    """Serialize / deserialize Episode objects to JSON."""

    # ------------------------------------------------------------------
    # Episode -> dict
    # ------------------------------------------------------------------

    def episode_to_dict(self, episode: Episode) -> Dict[str, Any]:
        meta = dict(episode.meta or {})
        meta.setdefault("schema_version", SCHEMA_VERSION)
        meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        return {
            "episode_id": episode.episode_id,
            "dataset": episode.dataset,
            "split": episode.split,
            "original_id": episode.original_id,
            "hop_count": episode.hop_count,
            "graph_type": episode.graph_type.value if isinstance(episode.graph_type, GraphType) else str(episode.graph_type),
            "reasoning_path_type": episode.reasoning_path_type.value if isinstance(episode.reasoning_path_type, ReasoningPathType) else str(episode.reasoning_path_type),
            "rho": episode.rho,
            "rho_subset": episode.rho_subset.value if isinstance(episode.rho_subset, SubsetLabel) else str(episode.rho_subset),
            "policy_conflict": episode.policy_conflict,
            "s4_eligible": episode.s4_eligible,
            "question": episode.question,
            "answer": episode.answer,
            "got_graph": episode.got_graph.to_dict(),
            "agents": [self._agent_to_dict(a) for a in episode.agents],
            "memory_entries": [self._memory_entry_to_dict(m) for m in episode.memory_entries],
            "retrieval_requests": [
                RetrievalRequestModel.from_request(r).model_dump() for r in episode.retrieval_requests
            ],
            "ground_truth": [g.to_dict() for g in episode.ground_truth],
            "meta": meta,
        }

    def episode_from_dict(self, data: Dict[str, Any]) -> Episode:
        got = GoTGraph.from_dict(data.get("got_graph", {}))
        graph_type = self._enum(data.get("graph_type"), GraphType, GraphType.LINEAR)
        reasoning = self._enum(data.get("reasoning_path_type"), ReasoningPathType, ReasoningPathType.GOT)
        rho_subset = self._enum(data.get("rho_subset"), SubsetLabel, SubsetLabel.S3)
        agents = [self._agent_from_dict(a) for a in data.get("agents", [])]
        memory_entries = [MemoryEntry.from_dict(m) for m in data.get("memory_entries", [])]
        retrieval_requests = [
            self._retrieval_request_from_dict(r) for r in data.get("retrieval_requests", [])
        ]
        ground_truth = [GroundTruthEvidence.from_dict(g) for g in data.get("ground_truth", [])]
        return Episode(
            episode_id=data["episode_id"],
            dataset=data["dataset"],
            split=data["split"],
            original_id=data["original_id"],
            hop_count=int(data.get("hop_count", 0)),
            graph_type=graph_type,
            reasoning_path_type=reasoning,
            rho=float(data.get("rho", 0.0)),
            rho_subset=rho_subset,
            policy_conflict=bool(data.get("policy_conflict", False)),
            s4_eligible=bool(data.get("s4_eligible", False)),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            got_graph=got,
            agents=agents,
            memory_entries=memory_entries,
            retrieval_requests=retrieval_requests,
            ground_truth=ground_truth,
            meta=data.get("meta", {}),
        )

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def write_jsonl(self, path: Union[str, Path], episodes: Iterable[Episode]) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(self.episode_to_dict(ep), ensure_ascii=False))
                f.write("\n")
                count += 1
        return count

    def append_jsonl(self, path: Union[str, Path], episodes: Iterable[Episode]) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "a", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(self.episode_to_dict(ep), ensure_ascii=False))
                f.write("\n")
                count += 1
        return count

    def read_jsonl(self, path: Union[str, Path]) -> List[Episode]:
        path = Path(path)
        if not path.exists():
            return []
        out: List[Episode] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(self.episode_from_dict(json.loads(line)))
        return out

    # ------------------------------------------------------------------
    # Agent / MemoryEntry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_to_dict(agent: Agent) -> Dict[str, Any]:
        cfg = agent.config
        return {
            "agent_id": cfg.agent_id,
            "role": cfg.role.value,
            "name": cfg.name,
            "description": cfg.description,
            "allowed_scopes": list(cfg.allowed_scopes),
            "allowed_memory_types": [t.value for t in cfg.allowed_memory_types],
            "policy": {
                "visibility": [v.value for v in cfg.policy.visibility],
                "max_clearance": cfg.policy.max_clearance,
                "excluded_zones": list(cfg.policy.excluded_zones),
                "audit_required": cfg.policy.audit_required,
            },
            "system_prompt": cfg.system_prompt,
            "metadata": dict(cfg.metadata),
            "state": agent.state.to_dict(),
        }

    @staticmethod
    def _agent_from_dict(data: Dict[str, Any]) -> Agent:
        policy = policy_constraints_from_dict(data.get("policy", {}))
        allowed_memory_types = [
            MemoryType(t) if isinstance(t, str) else t
            for t in data.get("allowed_memory_types", [])
        ]
        config = AgentConfig(
            agent_id=data["agent_id"],
            role=AgentRole(data.get("role", AgentRole.CUSTOM.value)),
            name=data.get("name", ""),
            description=data.get("description", ""),
            allowed_scopes=list(data.get("allowed_scopes", [])),
            allowed_memory_types=allowed_memory_types,
            policy=policy,
            system_prompt=data.get("system_prompt", ""),
            metadata=dict(data.get("metadata", {})),
        )
        state = AgentState.from_dict(data.get("state", {}))
        return Agent(config=config, state=state)

    @staticmethod
    def _memory_entry_to_dict(entry: MemoryEntry) -> Dict[str, Any]:
        d = entry.to_dict()
        d["embedding"] = None   # always strip embedding from the on-disk form
        return d

    @staticmethod
    def _retrieval_request_from_dict(data: Dict[str, Any]) -> RetrievalRequest:
        scope_data = data.get("scope", {})
        scope = ScopeSpec(
            private_scopes=list(scope_data.get("private_scopes", [])),
            shared_scopes=list(scope_data.get("shared_scopes", [])),
            workspace_scopes=list(scope_data.get("workspace_scopes", [])),
            governed_scopes=list(scope_data.get("governed_scopes", [])),
        )
        memory_types = [
            MemoryType(t) if isinstance(t, str) else t
            for t in data.get("memory_types", [])
        ]
        policy = policy_constraints_from_dict(data.get("policy", {}))
        state = AgentState.from_dict(data.get("state", {}))
        return RetrievalRequest(
            request_id=data.get("request_id", f"req_{data.get('agent_id', 'unknown')}"),
            agent_id=data.get("agent_id", ""),
            role=AgentRole(data.get("role", AgentRole.CUSTOM.value)),
            query=data.get("query", ""),
            scope=scope,
            memory_types=memory_types,
            policy=policy,
            state=state,
            priority=int(data.get("priority", 0)),
            metadata=dict(data.get("metadata", {})),
        )

    @staticmethod
    def _enum(value, enum_cls, default):
        if value is None:
            return default
        try:
            return enum_cls(value)
        except ValueError:
            return default
