"""
ArtifactRolloutEngine skeleton.

Runs a deterministic (given a seed) LLM role-play over a GoT graph and emits
an ArtifactTrace. The graph topology itself is NOT produced here; it is taken
as input from `data/got/graph_templates.py`.

This file is intentionally a stub: method bodies raise NotImplementedError.
A follow-up change implements them once the design is signed off.
See `coscope/data/docs/SystemLayer_Rollout_Design_v1.md` §7.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from coscope.core.types import GoTGraph, GoTNode, GraphType, NodeType, ReasoningPathType
from coscope.rollout import fallback_synth, prompt_assembly
from coscope.core.artifact_types import ArtifactSlot, ArtifactTrace
from coscope.rollout.artifact_writer import write_artifact
from coscope.core.interfaces import LLMClient
from coscope.rollout.trace_validator import validate_trace

logger = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    """All knobs that influence artifact generation (routed via main.py)."""

    temperature: float = 0.3
    top_p: float = 0.9
    seed: int = 42
    max_retries: int = 2
    max_new_tokens: int = 512
    use_template_fallback: bool = True
    emit_session_layer: bool = False   # v1 keeps session artifacts off by default


class ArtifactRolloutEngine:
    """
    Orchestrates per-episode rollout over a GoTGraph.

    Public surface mirrors §7 of the design doc. Internals (prompt assembly,
    parent artifact tracking, retry + fallback, trace validation) are kept as
    separate modules (`prompt_assembly`, `artifact_writer`, `fallback_synth`,
    `trace_validator`) so each step can be unit-tested and swapped.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        config: Optional[RolloutConfig] = None,
    ):
        self.llm_client = llm_client
        self.config = config or RolloutConfig()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        raw_item: Dict[str, Any],
        got_graph: GoTGraph,
        dataset: str,
        episode_id: str,
        reasoning_path_type: ReasoningPathType = ReasoningPathType.GOT,
    ) -> ArtifactTrace:
        """Execute the rollout and return an ArtifactTrace. See design doc §7."""
        trace = ArtifactTrace(
            episode_id=episode_id,
            rollout_meta={
                "engine": "ArtifactRolloutEngine/v1",
                "llm": {"name": getattr(self.llm_client, "name", "?")},
                "reasoning_path_type": reasoning_path_type.value,
                "seed": self.config.seed,
            },
        )
        topo_counter = {"v": 0}

        # --- 1. Planner -----------------------------------------------
        planner = got_graph.planner_node()
        planner_plan_id: Optional[str] = None
        planner_conclusion_id: Optional[str] = None  # planner has no conclusion; plan IS the outward artifact
        if planner is not None:
            self._emit_query_intent(
                trace, planner, role="planner", raw_item=raw_item,
                prior_conclusions=[], dataset=dataset, episode_id=episode_id,
                parent_ids=[], topo_counter=topo_counter,
            )
            self._emit_scratch(
                trace, planner, role="planner", raw_item=raw_item,
                prior_conclusions=[], dataset=dataset, episode_id=episode_id,
                parent_ids=[], topo_counter=topo_counter,
            )
            planner_plan_id = self._emit_plan(
                trace, planner, raw_item=raw_item,
                dataset=dataset, episode_id=episode_id,
                parent_ids=[], topo_counter=topo_counter,
            )

        # --- 2. Solvers in topological order --------------------------
        solver_nodes = self._topo_sorted_solvers(got_graph)
        node_to_conclusion_id: Dict[str, str] = {}

        for node in solver_nodes:
            # Parents may be planner or other solvers. Resolve to conclusion/plan ids.
            parent_ids: List[str] = []
            prior_conclusions: List[str] = []
            for p_node_id in node.parent_node_ids:
                if planner is not None and p_node_id == planner.node_id and planner_plan_id:
                    parent_ids.append(planner_plan_id)
                elif p_node_id in node_to_conclusion_id:
                    pc_id = node_to_conclusion_id[p_node_id]
                    parent_ids.append(pc_id)
                    # Retrieve content for prompt context
                    for e in trace.entries:
                        if e.memory_id == pc_id:
                            prior_conclusions.append(e.content)
                            break

            self._emit_query_intent(
                trace, node, role="solver", raw_item=raw_item,
                prior_conclusions=prior_conclusions,
                dataset=dataset, episode_id=episode_id,
                parent_ids=parent_ids, topo_counter=topo_counter,
            )
            self._emit_scratch(
                trace, node, role="solver", raw_item=raw_item,
                prior_conclusions=prior_conclusions,
                dataset=dataset, episode_id=episode_id,
                parent_ids=parent_ids, topo_counter=topo_counter,
            )
            concl_id = self._emit_conclusion(
                trace, node, raw_item=raw_item,
                prior_conclusions=prior_conclusions,
                dataset=dataset, episode_id=episode_id,
                parent_ids=parent_ids, topo_counter=topo_counter,
            )
            node_to_conclusion_id[node.node_id] = concl_id

        # --- 3. Verifier (S4 only) ------------------------------------
        verifier = got_graph.verifier_node()
        if verifier is not None and got_graph.graph_type == GraphType.POLICY_ISOLATED:
            all_conclusions = [
                e.content for e in trace.entries
                if (e.metadata or {}).get("slot") == ArtifactSlot.CONCLUSION.value
            ]
            all_conclusion_ids = [
                e.memory_id for e in trace.entries
                if (e.metadata or {}).get("slot") == ArtifactSlot.CONCLUSION.value
            ]
            self._emit_query_intent(
                trace, verifier, role="verifier", raw_item=raw_item,
                prior_conclusions=all_conclusions, dataset=dataset,
                episode_id=episode_id, parent_ids=all_conclusion_ids,
                topo_counter=topo_counter,
            )
            self._emit_scratch(
                trace, verifier, role="verifier", raw_item=raw_item,
                prior_conclusions=all_conclusions, dataset=dataset,
                episode_id=episode_id, parent_ids=all_conclusion_ids,
                topo_counter=topo_counter,
            )
            self._emit_audit_report(
                trace, verifier, raw_item=raw_item,
                all_conclusions=all_conclusions,
                dataset=dataset, episode_id=episode_id,
                parent_ids=all_conclusion_ids, topo_counter=topo_counter,
            )

        # --- 4. Post hoc: populate consumer_index ---------------------
        trace.consumer_index = self._compute_consumer_index(trace, got_graph)

        # --- 5. Validate trace ----------------------------------------
        result = validate_trace(trace, got_graph)
        if not result.passed:
            logger.warning(
                "Trace validation failed for %s: %s", episode_id, "; ".join(result.errors)
            )
        for w in result.warnings:
            logger.debug("Trace warning %s: %s", episode_id, w)

        return trace

    # ------------------------------------------------------------------
    # Internals: topological ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _topo_sorted_solvers(graph: GoTGraph) -> List[GoTNode]:
        """Kahn's algorithm over solver nodes only (skip planner/verifier)."""
        solvers = graph.solver_nodes()
        ids = {n.node_id for n in solvers}
        indeg: Dict[str, int] = {n.node_id: 0 for n in solvers}
        for n in solvers:
            for p in n.parent_node_ids:
                if p in ids:
                    indeg[n.node_id] += 1
        queue = deque([n for n in solvers if indeg[n.node_id] == 0])
        out: List[GoTNode] = []
        id_to_node = {n.node_id: n for n in solvers}
        while queue:
            cur = queue.popleft()
            out.append(cur)
            for ch in cur.child_node_ids:
                if ch in indeg:
                    indeg[ch] -= 1
                    if indeg[ch] == 0:
                        queue.append(id_to_node[ch])
        if len(out) != len(solvers):
            logger.warning("Solver topo-sort found cycle or missing edges; falling back to input order")
            return solvers
        return out

    # ------------------------------------------------------------------
    # Consumer index (who needs to read what, for rho computation)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_consumer_index(
        trace: ArtifactTrace, graph: GoTGraph
    ) -> Dict[str, List[str]]:
        """Map node_id -> memory_ids it structurally needs to consume."""
        slot_index: Dict[str, Dict[str, str]] = {}
        for e in trace.entries:
            meta = e.metadata or {}
            slot = meta.get("slot", "")
            node_id = meta.get("source_node_id", "")
            slot_index.setdefault(slot, {})[node_id] = e.memory_id

        consumer: Dict[str, List[str]] = {}
        plan_id = None
        planner = graph.planner_node()
        if planner is not None:
            plan_id = slot_index.get(ArtifactSlot.PLAN.value, {}).get(planner.node_id)

        for node in graph.solver_nodes():
            needs: List[str] = []
            if plan_id:
                needs.append(plan_id)
            # Ancestor solver conclusions
            for anc in graph.get_all_ancestors(node.node_id):
                c = slot_index.get(ArtifactSlot.CONCLUSION.value, {}).get(anc)
                if c:
                    needs.append(c)
            consumer[node.node_id] = needs
        return consumer

    # ------------------------------------------------------------------
    # Emit helpers: one per slot. Each writes a MemoryEntry into trace.
    # ------------------------------------------------------------------

    def _emit_query_intent(
        self, trace: ArtifactTrace, node: GoTNode, *,
        role: str, raw_item: Dict[str, Any],
        prior_conclusions: List[str], dataset: str, episode_id: str,
        parent_ids: List[str], topo_counter: Dict[str, int],
    ) -> str:
        prompt = prompt_assembly.build_query_intent_prompt(
            role=role, raw_item=raw_item, node=node,
            prior_conclusions=prior_conclusions,
            reasoning_path_type=ReasoningPathType.GOT,
        )
        content, source = self._call_llm_or_fallback(
            prompt,
            fallback_fn=lambda: {
                "planner": fallback_synth.synth_planner_query_intent(raw_item),
                "solver": fallback_synth.synth_solver_query_intent(raw_item, node),
                "verifier": fallback_synth.synth_verifier_query_intent(raw_item),
            }.get(role, ""),
        )
        return self._append_artifact(
            trace, slot=ArtifactSlot.QUERY_INTENT, node=node, content=content,
            dataset=dataset, episode_id=episode_id, parent_ids=parent_ids,
            source=source, topo_counter=topo_counter,
        )

    def _emit_scratch(
        self, trace: ArtifactTrace, node: GoTNode, *,
        role: str, raw_item: Dict[str, Any],
        prior_conclusions: List[str], dataset: str, episode_id: str,
        parent_ids: List[str], topo_counter: Dict[str, int],
    ) -> str:
        prompt = prompt_assembly.build_scratch_prompt(
            role=role, raw_item=raw_item, node=node,
            prior_conclusions=prior_conclusions,
            reasoning_path_type=ReasoningPathType.GOT,
        )
        content, source = self._call_llm_or_fallback(
            prompt,
            fallback_fn=lambda: {
                "planner": fallback_synth.synth_planner_scratch(raw_item),
                "solver": fallback_synth.synth_solver_scratch(raw_item, node),
                "verifier": fallback_synth.synth_verifier_scratch(raw_item),
            }.get(role, ""),
        )
        return self._append_artifact(
            trace, slot=ArtifactSlot.SCRATCH, node=node, content=content,
            dataset=dataset, episode_id=episode_id, parent_ids=parent_ids,
            source=source, topo_counter=topo_counter,
        )

    def _emit_plan(
        self, trace: ArtifactTrace, node: GoTNode, *,
        raw_item: Dict[str, Any], dataset: str, episode_id: str,
        parent_ids: List[str], topo_counter: Dict[str, int],
    ) -> str:
        prompt = prompt_assembly.build_planner_prompt(
            raw_item=raw_item, node=node, reasoning_path_type=ReasoningPathType.GOT,
        )
        content, source = self._call_llm_or_fallback(
            prompt, fallback_fn=lambda: fallback_synth.synth_plan(raw_item),
        )
        return self._append_artifact(
            trace, slot=ArtifactSlot.PLAN, node=node, content=content,
            dataset=dataset, episode_id=episode_id, parent_ids=parent_ids,
            source=source, topo_counter=topo_counter,
        )

    def _emit_conclusion(
        self, trace: ArtifactTrace, node: GoTNode, *,
        raw_item: Dict[str, Any], prior_conclusions: List[str],
        dataset: str, episode_id: str,
        parent_ids: List[str], topo_counter: Dict[str, int],
    ) -> str:
        prompt = prompt_assembly.build_solver_prompt(
            raw_item=raw_item, node=node, prior_conclusions=prior_conclusions,
            reasoning_path_type=ReasoningPathType.GOT,
        )
        content, source = self._call_llm_or_fallback(
            prompt, fallback_fn=lambda: fallback_synth.synth_solver_conclusion(raw_item, node),
        )
        # Mark as gold-evidence-derived when template fallback used gold oracle
        is_gold = source == "template" and bool(
            fallback_synth._task_shared_text_for_hop(raw_item, node.hop_index or 0)
        )
        return self._append_artifact(
            trace, slot=ArtifactSlot.CONCLUSION, node=node, content=content,
            dataset=dataset, episode_id=episode_id, parent_ids=parent_ids,
            source=source, topo_counter=topo_counter, is_gold_evidence=is_gold,
        )

    def _emit_audit_report(
        self, trace: ArtifactTrace, node: GoTNode, *,
        raw_item: Dict[str, Any], all_conclusions: List[str],
        dataset: str, episode_id: str,
        parent_ids: List[str], topo_counter: Dict[str, int],
    ) -> str:
        prompt = prompt_assembly.build_verifier_prompt(
            raw_item=raw_item, node=node, all_conclusions=all_conclusions,
            dataset=dataset, reasoning_path_type=ReasoningPathType.GOT,
        )
        content, source = self._call_llm_or_fallback(
            prompt,
            fallback_fn=lambda: fallback_synth.synth_audit_report(
                raw_item, all_conclusions, dataset
            ),
        )
        return self._append_artifact(
            trace, slot=ArtifactSlot.AUDIT_REPORT, node=node, content=content,
            dataset=dataset, episode_id=episode_id, parent_ids=parent_ids,
            source=source, topo_counter=topo_counter, required_clearance=3,
        )

    # ------------------------------------------------------------------
    # LLM call with template fallback
    # ------------------------------------------------------------------

    def _call_llm_or_fallback(self, prompt: str, *, fallback_fn) -> tuple:
        """Return (content, source_tag). source_tag is 'llm' or 'template'."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.llm_client.generate(
                    prompt,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    seed=self.config.seed,
                    max_new_tokens=self.config.max_new_tokens,
                )
                text = (resp.text or "").strip()
                if text:
                    return text, "llm"
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.debug("LLM call failed (attempt %d): %s", attempt, exc)
        if self.config.use_template_fallback:
            return fallback_fn() or "", "template"
        raise RuntimeError(f"LLM call failed after retries: {last_exc}")

    # ------------------------------------------------------------------
    # Append helper
    # ------------------------------------------------------------------

    def _append_artifact(
        self, trace: ArtifactTrace, *,
        slot: ArtifactSlot, node: GoTNode, content: str,
        dataset: str, episode_id: str, parent_ids: List[str],
        source: str, topo_counter: Dict[str, int],
        is_gold_evidence: bool = False, required_clearance: int = 0,
    ) -> str:
        idx = topo_counter["v"]
        topo_counter["v"] += 1
        llm_name = getattr(self.llm_client, "name", None) if source == "llm" else None
        entry = write_artifact(
            slot=slot, node=node, content=content,
            dataset=dataset, episode_id=episode_id,
            topo_index=idx, parent_artifact_ids=parent_ids,
            is_gold_evidence=is_gold_evidence,
            required_clearance=required_clearance,
            llm_name=llm_name,
        )
        trace.entries.append(entry)
        trace.producer_index.setdefault(node.node_id, []).append(entry.memory_id)
        trace.stream.append((idx, entry.memory_id))
        return entry.memory_id
