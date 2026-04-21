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
from coscope.data.core.types import Episode, GraphType
from coscope.data.memory.scope_ids import task_restricted

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
        self._check_workspace(episode, result)
        self._check_agents(episode, result)
        self._check_restricted(episode, result)
        self._check_memory_ids_unique(episode, result)
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
        if "workspace_semantic_global" not in layers:
            result.add_error("workspace_semantic_global layer is empty")
        # hop-level workspace is optional for graphs whose solvers lack hop_index

    def _check_agents(self, ep: Episode, result: ValidationResult) -> None:
        restricted_scope = task_restricted(ep.episode_id)
        has_verifier = False
        for agent in ep.agents:
            role = agent.config.role
            scopes = set(agent.config.allowed_scopes)
            clearance = int(agent.config.policy.max_clearance or 0)
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
                "workspace_semantic_global",
                "workspace_semantic_hop",
                "task_shared_episodic",
            ):
                continue
            if answer in (entry.content or ""):
                result.add_warning(
                    f"answer leak suspicion: '{answer}' found in public memory {entry.memory_id}"
                )
