"""
End-to-end rollout demo.

Runs ArtifactRolloutEngine on a handful of MuSiQue raw items under each GoT
graph type, validates the resulting traces, and prints rho distribution. Use
this as the sanity check for schema and runtime logic before scaling up.

Run:
    python -m coscope.scripts.demo_rollout
or route via main.py.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

from coscope.core.types import GraphType, ReasoningPathType
from coscope.graph.got.graph_builder import GraphBuilder
from coscope.io.loaders import get_loader
from coscope.rollout import ArtifactRolloutEngine
from coscope.rollout.rho_v3 import compute_rho
from coscope.rollout.rollout_engine import RolloutConfig
from coscope.llm.template import TemplateLLMClient
from coscope.rollout.trace_validator import validate_trace


def _format_solver_pair_iou(trace, got_graph) -> str:
    """Small debug helper: show pairwise IoU for each solver pair."""
    from coscope.core.artifact_types import ArtifactSlot
    by_slot = {}
    for e in trace.entries:
        meta = e.metadata or {}
        by_slot.setdefault(meta.get("slot"), {})[meta.get("source_node_id")] = e.memory_id
    plan_ids = set(by_slot.get(ArtifactSlot.PLAN.value, {}).values())
    concl = by_slot.get(ArtifactSlot.CONCLUSION.value, {})

    solver_sets = {}
    for node in got_graph.solver_nodes():
        anc = set(got_graph.get_all_ancestors(node.node_id)) | {node.node_id}
        solver_sets[node.node_id] = set(plan_ids) | {
            concl[n] for n in anc if n in concl
        }
    lines = []
    names = list(solver_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = solver_sets[names[i]], solver_sets[names[j]]
            iou = len(a & b) / max(1, len(a | b))
            lines.append(f"    {names[i]} vs {names[j]}: {iou:.3f} "
                         f"(|A|={len(a)}, |B|={len(b)}, |A∩B|={len(a & b)})")
    return "\n".join(lines) if lines else "    (only one solver — degenerate)"


def run_demo(
    dataset: str = "musique",
    split: str = "dev",
    n_items: int = 3,
    graph_types: List[GraphType] = None,
    use_llm: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    graph_types = graph_types or [
        GraphType.LINEAR, GraphType.FORK, GraphType.FORK_MERGE,
        GraphType.INDEPENDENT, GraphType.POLICY_ISOLATED,
    ]

    loader = get_loader(dataset, data_dir=f"coscope/data/raw/{dataset}", split=split)
    raw_items = list(loader.load())[:n_items]
    if not raw_items:
        print(f"No raw items for {dataset}/{split}; aborting.")
        return

    # TemplateLLMClient -> deterministic. Swap to QwenClient later.
    llm_client = TemplateLLMClient()
    engine = ArtifactRolloutEngine(llm_client, config=RolloutConfig(seed=42))
    graph_builder = GraphBuilder()

    rho_by_gtype: dict = defaultdict(list)

    for gt in graph_types:
        print(f"\n=== graph_type = {gt.value} " + "=" * 40)
        for idx, raw in enumerate(raw_items):
            original_id = str(raw.get("original_id", f"item_{idx}"))
            episode_id = f"demo_{dataset}_{split}_{original_id}_{gt.value}"
            try:
                got_graph = graph_builder.build(
                    raw, dataset=dataset, target_graph_type=gt, seed=42
                )
            except Exception as exc:
                print(f"  [skip] {original_id}: graph build failed: {exc}")
                continue

            trace = engine.run(
                raw_item=raw, got_graph=got_graph,
                dataset=dataset, episode_id=episode_id,
                reasoning_path_type=ReasoningPathType.GOT,
            )
            result = validate_trace(trace, got_graph)
            rho = compute_rho(trace, got_graph)
            rho_by_gtype[gt.value].append(rho)

            n_solvers = len(got_graph.solver_nodes())
            n_artifacts = len(trace.entries)
            print(f"  [{idx + 1}/{len(raw_items)}] {original_id[:40]:40s} "
                  f"solvers={n_solvers} artifacts={n_artifacts} rho={rho:.3f} "
                  f"valid={'OK' if result.passed else 'FAIL'}")
            if not result.passed:
                for err in result.errors[:3]:
                    print(f"      ERROR: {err}")
            if n_solvers >= 2:
                print(_format_solver_pair_iou(trace, got_graph))

    print("\n=== rho summary " + "=" * 48)
    for gt_name, rhos in rho_by_gtype.items():
        if not rhos:
            continue
        avg = sum(rhos) / len(rhos)
        print(f"  {gt_name:20s} n={len(rhos)} mean_rho={avg:.3f} values={[round(r, 3) for r in rhos]}")


if __name__ == "__main__":
    run_demo()
