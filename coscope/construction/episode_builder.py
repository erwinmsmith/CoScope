"""
Single-episode builder.

Orchestrates graph construction, memory store construction, agent generation,
rho computation, subset labeling, and static validation.

NOTE on `episode_id` composition
--------------------------------
The spec (§11.4) suggests appending `_{rho_subset}` to `episode_id`, but rho is
not known until after memory + agents are built (and those construction steps
need a stable `episode_id` for scope_id / agent_id generation). We therefore
use a stable stem `{dataset}_{split}_{original_id}_{graph_type}` as `episode_id`
and expose `rho_subset` as a separate field. The sharding code in
DatasetPipeline still writes to the right per-subset file.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from coscope.core.types import Agent, MemoryEntry, RetrievalRequest
from coscope.agents import (
    PlannerAgentBuilder,
    SolverAgentBuilder,
    VerifierAgentBuilder,
)
from coscope.auth.policy_validator import PolicyValidator, ValidationResult
from coscope.core.types import (
    Episode,
    GoTGraph,
    GraphType,
    GroundTruthEvidence,
    InvalidGraphError,
    ReasoningPathType,
)
from coscope.graph.got.graph_builder import GraphBuilder
from coscope.graph.got.rho_calculator import RhoCalculator
from coscope.memory.private_builder import PrivateBuilder
from coscope.memory.restricted_builder import RestrictedBuilder
from coscope.memory.task_shared_builder import TaskSharedBuilder
from coscope.memory.workspace_builder import WorkspaceBuilder
from coscope.utils.split.subset_assigner import SubsetAssigner, load_default_thresholds


logger = logging.getLogger(__name__)


_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(s: str) -> str:
    return _ID_SAFE_RE.sub("_", s).strip("_")


class EpisodeBuilder:
    """Build a single Episode from a raw_item + target graph_type."""

    def __init__(
        self,
        *,
        graph_builder: Optional[GraphBuilder] = None,
        workspace_builder: Optional[WorkspaceBuilder] = None,
        task_shared_builder: Optional[TaskSharedBuilder] = None,
        private_builder: Optional[PrivateBuilder] = None,
        restricted_builder: Optional[RestrictedBuilder] = None,
        rho_calculator: Optional[RhoCalculator] = None,
        subset_assigner: Optional[SubsetAssigner] = None,
        policy_validator: Optional[PolicyValidator] = None,
        planner_builder: Optional[PlannerAgentBuilder] = None,
        solver_builder: Optional[SolverAgentBuilder] = None,
        verifier_builder: Optional[VerifierAgentBuilder] = None,
    ):
        self.graph_builder = graph_builder or GraphBuilder()
        self.workspace_builder = workspace_builder or WorkspaceBuilder()
        self.task_shared_builder = task_shared_builder or TaskSharedBuilder()
        self.private_builder = private_builder or PrivateBuilder()
        self.restricted_builder = restricted_builder or RestrictedBuilder()
        self.rho_calculator = rho_calculator or RhoCalculator()
        self.subset_assigner = subset_assigner or SubsetAssigner(
            thresholds=load_default_thresholds()
        )
        self.policy_validator = policy_validator or PolicyValidator()
        self.planner_builder = planner_builder or PlannerAgentBuilder()
        self.solver_builder = solver_builder or SolverAgentBuilder()
        self.verifier_builder = verifier_builder or VerifierAgentBuilder()

    # ------------------------------------------------------------------

    def build_episode(
        self,
        raw_item: Dict[str, Any],
        dataset: str,
        split: str,
        target_graph_type: Union[str, GraphType],
        seed: int = 42,
    ) -> Optional[Episode]:
        try:
            got_graph = self.graph_builder.build(
                raw_item, dataset=dataset, target_graph_type=target_graph_type, seed=seed
            )
        except InvalidGraphError as exc:
            logger.warning("Skipping episode: %s", exc)
            return None

        graph_type = got_graph.graph_type
        original_id = str(raw_item.get("original_id", ""))
        episode_id = self._compose_episode_id(dataset, split, original_id, graph_type)

        # --- Memory store construction -------------------------------------
        workspace_entries = self.workspace_builder.build(raw_item, dataset, episode_id)
        task_shared_entries = self.task_shared_builder.build(
            raw_item, got_graph, episode_id, dataset
        )
        private_entries = self.private_builder.build(got_graph, episode_id, dataset)
        restricted_entries: List[MemoryEntry] = []
        if graph_type == GraphType.POLICY_ISOLATED:
            restricted_entries = self.restricted_builder.load_from_interim(
                dataset, original_id, episode_id
            )

        memory_entries: List[MemoryEntry] = (
            list(workspace_entries)
            + list(task_shared_entries)
            + list(private_entries)
            + list(restricted_entries)
        )

        # --- Agents + retrieval requests -----------------------------------
        agents: List[Agent] = []
        retrieval_requests: List[RetrievalRequest] = []

        planner_agent, planner_req = self.planner_builder.build(
            raw_item, got_graph, episode_id, dataset
        )
        agents.append(planner_agent)
        retrieval_requests.append(planner_req)

        for solver_agent, solver_req in self.solver_builder.build_all(
            raw_item, got_graph, episode_id, dataset
        ):
            agents.append(solver_agent)
            retrieval_requests.append(solver_req)

        verifier_result = self.verifier_builder.build(raw_item, got_graph, episode_id, dataset)
        if verifier_result is not None:
            agents.append(verifier_result[0])
            retrieval_requests.append(verifier_result[1])

        # --- rho + subset --------------------------------------------------
        rho = self.rho_calculator.compute(
            got_graph=got_graph,
            memory_entries=memory_entries,
            dataset=dataset,
            episode_id=episode_id,
            use_cache=False,  # caller may call flush() separately if using cache
        )
        assignment = self.subset_assigner.assign(rho, got_graph)

        # --- Ground truth --------------------------------------------------
        ground_truth = self._build_ground_truth(memory_entries, agents, got_graph)

        episode = Episode(
            episode_id=episode_id,
            dataset=dataset,
            split=split,
            original_id=original_id,
            hop_count=int(raw_item.get("hop_count", 0)),
            graph_type=graph_type,
            reasoning_path_type=ReasoningPathType.GOT,
            rho=rho,
            rho_subset=assignment.rho_subset,
            policy_conflict=assignment.policy_conflict,
            s4_eligible=assignment.s4_eligible,
            question=str(raw_item.get("question", "")),
            answer=str(raw_item.get("answer", "")),
            got_graph=got_graph,
            agents=agents,
            memory_entries=memory_entries,
            retrieval_requests=retrieval_requests,
            ground_truth=ground_truth,
            meta={
                "schema_version": "1.0.0",
                "seed": int(seed),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "reasoning_path_type": ReasoningPathType.GOT.value,
            },
        )

        # --- Static validation ---------------------------------------------
        validation: ValidationResult = self.policy_validator.validate(episode)
        if not validation.passed:
            logger.warning(
                "Discarding episode %s after policy validation: %s",
                episode_id,
                "; ".join(validation.errors),
            )
            return None
        for w in validation.warnings:
            logger.info("episode %s warning: %s", episode_id, w)
        return episode

    # ------------------------------------------------------------------

    @staticmethod
    def _compose_episode_id(
        dataset: str, split: str, original_id: str, graph_type: GraphType
    ) -> str:
        return f"{_slug(dataset)}_{_slug(split)}_{_slug(original_id)}_{graph_type.value}"

    @staticmethod
    def _build_ground_truth(
        memory_entries: List[MemoryEntry],
        agents: List[Agent],
        got_graph: GoTGraph,
    ) -> List[GroundTruthEvidence]:
        # Map source_node_id -> set of agent_ids that need it (agent covers ancestors+self).
        node_to_agents: Dict[str, List[str]] = {}
        for agent in agents:
            node_id = (agent.config.metadata or {}).get("node_id")
            if not node_id:
                continue
            accessible = (agent.config.metadata or {}).get("accessible_ancestor_node_ids") or []
            for anc in accessible:
                node_to_agents.setdefault(str(anc), []).append(agent.agent_id)

        ground_truth: List[GroundTruthEvidence] = []
        for entry in memory_entries:
            meta = entry.metadata or {}
            if not meta.get("is_gold_evidence"):
                continue
            layer = meta.get("scope_layer", "")
            if layer in ("workspace_semantic", "workspace_semantic_global"):
                # v3 flat workspace: every agent needs to recall gold paragraphs.
                required_by = [a.agent_id for a in agents]
                evidence_type = "shared_required"
                hop_index = meta.get("hop_index")
            elif layer == "task_shared_episodic":
                src = meta.get("source_node_id")
                required_by = node_to_agents.get(str(src), [])
                evidence_type = "shared_required"
                hop_index = meta.get("hop_index")
            else:
                continue
            ground_truth.append(
                GroundTruthEvidence(
                    memory_id=entry.memory_id,
                    scope_layer=layer,
                    hop_index=hop_index if isinstance(hop_index, int) else None,
                    required_by_agent_ids=required_by,
                    evidence_type=evidence_type,
                )
            )
        return ground_truth
