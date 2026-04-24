"""
Static policy validator.

Runs at episode construction time (not at runtime). Checks the invariants
described in spec §13.4 + §16.1 and returns a structured result. The caller
(EpisodeBuilder) is expected to discard episodes that fail any check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

from coscope.core.types import AgentRole
from coscope.core.types import Episode, GraphType
from coscope.core.scope_ids import agent_private, task_restricted

if TYPE_CHECKING:
    pass


@dataclass
class ValidationResult:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class PolicyValidator:
    """Episode-level static compliance checks."""

    def validate(self, episode: Episode) -> ValidationResult:
        result = ValidationResult(passed=True)
        self._check_graph(episode, result)
        self._check_episode_id(episode, result)
        self._check_workspace(episode, result)
        self._check_agents(episode, result)
        self._check_agent_count(episode, result)
        self._check_scope_isolation(episode, result)
        self._check_restricted(episode, result)
        self._check_memory_ids_unique(episode, result)
        self._check_embedding_null(episode, result)
        self._check_task_shared_init(episode, result)
        self._check_ground_truth(episode, result)
        self._check_rho(episode, result)
        self._check_answer_leak(episode, result)
        return result

    # ------------------------------------------------------------------

    def _check_graph(self, ep: Episode, result: ValidationResult) -> None:
        if ep.got_graph is None:
            result.add_error("got_graph is None")
            return
        try:
            ep.got_graph.validate()
        except Exception as exc:  # noqa: BLE001
            result.add_error(f"got_graph failed validation: {exc}")

    def _check_workspace(self, ep: Episode, result: ValidationResult) -> None:
        layers = {
            (e.metadata or {}).get("scope_layer", "") for e in ep.memory_entries
        }
        if "workspace_semantic" not in layers and "workspace_semantic_global" not in layers:
            result.add_error("workspace layer is empty (expected workspace_semantic or workspace_semantic_global)")
        # hop-level workspace is optional for graphs whose solvers lack hop_index

    def _check_agents(self, ep: Episode, result: ValidationResult) -> None:
        restricted_scope = task_restricted(ep.episode_id)
        has_verifier = False
        for agent in ep.agents:
            role = agent.config.role
            scopes = set(agent.config.allowed_scopes)
            clearance = int(agent.config.policy.max_clearance or 0)
            # §16.1(2) every agent must have at least one scope.
            if len(scopes) == 0:
                result.add_error(
                    f"agent {agent.agent_id} has empty allowed_scopes"
                )
            if role == AgentRole.VERIFIER:
                has_verifier = True
                if restricted_scope not in scopes:
                    result.add_error(
                        f"verifier agent {agent.agent_id} missing restricted scope"
                    )
                if clearance != 2:
                    result.add_error(
                        f"verifier agent {agent.agent_id} must have max_clearance=2 "
                        f"(got {clearance})"
                    )
            else:
                if restricted_scope in scopes:
                    result.add_error(
                        f"non-verifier agent {agent.agent_id} must NOT include restricted scope"
                    )
                if clearance != 1:
                    result.add_error(
                        f"non-verifier agent {agent.agent_id} must have max_clearance=1 "
                        f"(got {clearance})"
                    )

        # S4 eligibility must match Verifier presence.
        if ep.s4_eligible and not has_verifier:
            result.add_error("s4_eligible=True but no VERIFIER agent present")
        if ep.graph_type == GraphType.POLICY_ISOLATED and not has_verifier:
            result.add_error("POLICY_ISOLATED graph has no VERIFIER agent")
        # §16.1(2) Verifier only exists in POLICY_ISOLATED graphs.
        if has_verifier and ep.graph_type != GraphType.POLICY_ISOLATED:
            result.add_error(
                f"non-POLICY_ISOLATED graph (type={ep.graph_type}) must not contain VERIFIER"
            )

    def _check_restricted(self, ep: Episode, result: ValidationResult) -> None:
        for entry in ep.memory_entries:
            layer = (entry.metadata or {}).get("scope_layer", "")
            if layer != "restricted":
                continue
            required = int((entry.metadata or {}).get("required_clearance", 0) or 0)
            if required != 2:
                result.add_error(
                    f"restricted memory {entry.memory_id} has required_clearance={required} "
                    f"(expected 2)"
                )

    def _check_memory_ids_unique(self, ep: Episode, result: ValidationResult) -> None:
        seen = set()
        for entry in ep.memory_entries:
            if entry.memory_id in seen:
                result.add_error(f"duplicate memory_id: {entry.memory_id}")
            seen.add(entry.memory_id)

    def _check_rho(self, ep: Episode, result: ValidationResult) -> None:
        if not (0.0 <= ep.rho <= 1.0):
            result.add_error(f"rho out of range: {ep.rho}")

    def _check_answer_leak(self, ep: Episode, result: ValidationResult) -> None:
        answer = (ep.answer or "").strip()
        if not answer or len(answer) < 2:
            return
        for entry in ep.memory_entries:
            layer = (entry.metadata or {}).get("scope_layer", "")
            if layer not in (
                "workspace_semantic",
                "workspace_semantic_global",
                "workspace_semantic_hop",
                "task_shared_episodic",
            ):
                continue
            if answer in (entry.content or ""):
                result.add_warning(
                    f"answer leak suspicion: '{answer}' found in public memory {entry.memory_id}"
                )

    # ------------------------------------------------------------------
    # v2.2 §16.1 extended checks
    # ------------------------------------------------------------------

    def _check_episode_id(self, ep: Episode, result: ValidationResult) -> None:
        """§16.1(1): episode_id must follow `{dataset}_{split}_{original_id}_{graph_type}`."""
        reasoning = (
            ep.reasoning_path_type.value.lower()
            if hasattr(ep.reasoning_path_type, "value")
            else str(ep.reasoning_path_type).lower()
        )
        gt = ep.graph_type.value if isinstance(ep.graph_type, GraphType) else str(ep.graph_type)
        expected_suffix = f"_{reasoning}_{gt}"
        if not ep.episode_id.endswith(expected_suffix):
            result.add_error(
                f"episode_id {ep.episode_id!r} does not end with reasoning+graph "
                f"suffix {expected_suffix!r}"
            )
        if ep.dataset and not ep.episode_id.startswith(ep.dataset.replace("/", "_")):
            result.add_warning(
                f"episode_id {ep.episode_id!r} does not start with dataset "
                f"prefix {ep.dataset!r}"
            )

    def _check_agent_count(self, ep: Episode, result: ValidationResult) -> None:
        """§16.1(4): agent count >= hop_count + 1 (1 Planner + >= h Solvers)."""
        expected_min = int(ep.hop_count or 0) + 1
        if len(ep.agents) < expected_min:
            result.add_error(
                f"agent count {len(ep.agents)} < hop_count+1 ({expected_min}) "
                f"for episode {ep.episode_id}"
            )

    def _check_scope_isolation(self, ep: Episode, result: ValidationResult) -> None:
        """§16.1(2): a Solver's private scope must not appear in Planner's allowed_scopes."""
        planner_scopes: set = set()
        for agent in ep.agents:
            if agent.config.role == AgentRole.PLANNER:
                planner_scopes = set(agent.config.allowed_scopes)
                break
        for node in ep.got_graph.solver_nodes() if ep.got_graph else []:
            priv = agent_private(ep.episode_id, node.node_id)
            if priv in planner_scopes:
                result.add_error(
                    f"planner has access to solver-private scope {priv} "
                    f"(node={node.node_id})"
                )

    def _check_embedding_null(self, ep: Episode, result: ValidationResult) -> None:
        """§16.1(1): `embedding` must be null at construction time."""
        for entry in ep.memory_entries:
            emb = getattr(entry, "embedding", None)
            if emb is not None:
                result.add_error(
                    f"memory {entry.memory_id} has non-null embedding at construction"
                )

    def _check_task_shared_init(self, ep: Episode, result: ValidationResult) -> None:
        """
        §16.1(1): task-shared layer size sanity.

        NOTE: v2.2 spec describes the RUNTIME start-time state (0 items for QA
        under online mode; h items for math under oracle mode). At build time,
        our pipeline pre-fills task_shared with gold oracle labels so that
        static rho can be computed without running any agent. We therefore
        only emit warnings when the build-time count looks clearly wrong
        (e.g. exceeds hop_count * 2 for QA, or deviates from hop_count for
        math datasets).
        """
        shared = [
            e for e in ep.memory_entries
            if (e.metadata or {}).get("scope_layer") == "task_shared_episodic"
        ]
        ds = (ep.dataset or "").lower()
        qa_datasets = {"musique", "hotpotqa", "2wikimhqa", "2wikimultihopqa"}
        math_datasets = {"gsm8k", "math"}
        solver_count = len([n for n in (ep.got_graph.solver_nodes() if ep.got_graph else [])])
        if ds in qa_datasets:
            # Pre-filled oracle label: one entry per solver node is expected.
            if len(shared) > max(2 * solver_count, 2 * int(ep.hop_count or 0)):
                result.add_warning(
                    f"QA dataset {ds}: task_shared count {len(shared)} exceeds "
                    f"expected oracle-pre-fill bound (2 * solver_count={solver_count})"
                )
        elif ds in math_datasets:
            expected = int(ep.hop_count or 0)
            if expected > 0 and len(shared) not in (expected, 0):
                result.add_warning(
                    f"math dataset {ds}: task_shared count {len(shared)} "
                    f"!= hop_count {expected} (accepted: 0 or {expected})"
                )

    def _check_ground_truth(self, ep: Episode, result: ValidationResult) -> None:
        """
        §16.1(3): ground-truth evidence coverage and consistency.

        Hard checks (error if violated):
        - At least one ground-truth evidence per episode.
        - All gt.memory_id must exist in memory_entries.
        - Every non-Verifier agent must appear in `required_by_agent_ids`
          of at least one evidence.

        Soft checks (warning only):
        - `shared_required` evidences stored in a layer that ends up visible
          to only one build-time agent (e.g. workspace_semantic_global sees
          Planner only). This is expected for the layered architecture where
          Planner later distills into task_shared, but we warn if NONE of the
          `shared_required` evidences live in `task_shared_episodic`, since
          that would mean no evidence is team-visible at all.
        """
        if not ep.ground_truth:
            result.add_error(f"episode {ep.episode_id} has no ground-truth evidence")
            return

        mem_by_id = {e.memory_id: e for e in ep.memory_entries}
        required_set: set = set()
        shared_layers: list = []
        for gt in ep.ground_truth:
            if gt.memory_id not in mem_by_id:
                result.add_error(
                    f"ground-truth memory_id {gt.memory_id} not found in memory_entries"
                )
                continue
            required_set.update(gt.required_by_agent_ids)
            if gt.evidence_type == "shared_required":
                entry = mem_by_id[gt.memory_id]
                shared_layers.append((entry.metadata or {}).get("scope_layer", ""))

        # Warn if no team-visible (task_shared_episodic) shared_required item
        # exists — means nothing is cross-agent visible at build time.
        if shared_layers and "task_shared_episodic" not in shared_layers:
            result.add_warning(
                f"episode {ep.episode_id}: no shared_required evidence in "
                f"task_shared_episodic layer (layers seen: {sorted(set(shared_layers))})"
            )

        # Every non-Verifier agent must require at least one evidence.
        for agent in ep.agents:
            if agent.config.role == AgentRole.VERIFIER:
                continue
            if agent.agent_id not in required_set:
                result.add_error(
                    f"agent {agent.agent_id} has no ground-truth evidence assigned"
                )
