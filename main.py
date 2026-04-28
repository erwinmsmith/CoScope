"""
CoScope — unified CLI entry.

Single command that:
  1. loads raw items
  2. runs EpisodeBuilder (static: workspace + oracle task_shared + agents
     + ground_truth)
  3. runs ArtifactRolloutEngine (LLM rollout -> ArtifactTrace)
  4. merges trace into Episode.memory_entries
  5. recomputes rho from the *actual* rollout trace
  6. re-assigns S1/S2/S3 subset with fresh rho
  7. validates trace (8 hard rules)
  8. serializes to JSONL shards:
     data/processed/{reasoning_path}/{dataset}/{split}/{subset}_{gt}.jsonl

All hyper-parameters are CLI flags (no bash wrappers, per project convention).

Examples
--------
Small qwen-plus validation run:

    python main.py \\
        --datasets musique --splits dev --limit 3 \\
        --llm dashscope --llm-model qwen-plus

Full production (all datasets × all graph types):

    python main.py \\
        --datasets musique 2wikimhqa hotpotqa gsm8k math \\
        --splits train dev test \\
        --llm dashscope --llm-model qwen-plus \\
        --embedder dashscope
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv


logger = logging.getLogger(__name__)


# ============================================================
# Backend factories
# ============================================================


def _build_llm(name: str, model: str):
    if name == "template":
        from llm.template import TemplateLLMClient
        return TemplateLLMClient()
    if name == "dashscope":
        from llm.dashscope import DashScopeClient
        return DashScopeClient(model=model)
    raise ValueError(f"Unknown --llm backend: {name}")


def _build_embedder(name: str, model: str, dim: int):
    if name == "none":
        return None
    if name == "dashscope":
        from embedding.dashscope_embedder import DashScopeEmbedder
        return DashScopeEmbedder(model=model, dim=dim)
    raise ValueError(f"Unknown --embedder backend: {name}")


# ============================================================
# Core per-episode routine
# ============================================================


def _process_one(
    *,
    raw_item: dict,
    dataset: str,
    split: str,
    target_graph_type,
    seed: int,
    episode_builder,
    rollout_engine,
    subset_assigner,
    serializer,
):
    """Return (episode_dict, rho) or (None, reason) on failure."""
    from core.types import ReasoningPathType
    from rollout.rho_calculator import compute_rho
    from rollout.trace_validator import validate_trace

    # (1) static episode
    ep = episode_builder.build_episode(
        raw_item=raw_item, dataset=dataset, split=split,
        target_graph_type=target_graph_type, seed=seed,
    )
    if ep is None:
        return None, "static_build_failed"

    # (2) rollout
    trace = rollout_engine.run(
        raw_item=raw_item, got_graph=ep.got_graph,
        dataset=dataset, episode_id=ep.episode_id,
        reasoning_path_type=ep.reasoning_path_type,
    )
    tr_result = validate_trace(trace, ep.got_graph)
    if not tr_result.passed:
        return None, f"trace_invalid:{tr_result.errors[:2]}"

    # (3) merge, (4) rho, (5) subset
    rho = compute_rho(trace, ep.got_graph)
    assignment = subset_assigner.assign(rho, ep.got_graph)
    ep.rho = float(rho)
    ep.rho_subset = assignment.rho_subset
    ep.policy_conflict = assignment.policy_conflict
    ep.s4_eligible = assignment.s4_eligible
    ep.meta["rho_version"] = "v3"
    ep.memory_entries.extend(trace.entries)
    ep.meta["rollout"] = {
        "engine": "ArtifactRolloutEngine/v1",
        "n_entries": len(trace.entries),
        **(trace.rollout_meta or {}),
    }

    ep_dict = serializer.episode_to_dict(ep)
    return ep_dict, rho


# ============================================================
# Main build loop
# ============================================================


def run_build(args) -> int:
    from construction import EpisodeBuilder
    from core.types import GraphType
    from rollout import ArtifactRolloutEngine
    from rollout.rollout_engine import RolloutConfig
    from dataio.loaders import get_loader
    from dataio import Serializer
    from evaluation.split import SubsetAssigner, load_default_thresholds

    llm = _build_llm(args.llm, args.llm_model)
    embedder = _build_embedder(args.embedder, args.embedder_model, args.embedder_dim)
    logger.info("Backend: llm=%s embedder=%s seed=%d", llm.name,
                embedder.name if embedder else "none", args.seed)

    episode_builder = EpisodeBuilder(
        reasoning_path_type=args.reasoning_path_type,
    )
    engine = ArtifactRolloutEngine(
        llm,
        config=RolloutConfig(seed=args.seed, temperature=args.temperature,
                             top_p=args.top_p, max_new_tokens=args.max_new_tokens),
    )
    serializer = Serializer()
    subset_assigner = SubsetAssigner(thresholds=load_default_thresholds())

    total_ok = total_fail = 0
    total_rhos: Dict[str, List[float]] = defaultdict(list)
    shard_counts: Dict[str, int] = defaultdict(int)

    t_run_start = time.perf_counter()

    for dataset in args.datasets:
        for split in args.splits:
            loader = get_loader(dataset, data_dir=f"{args.data_dir}/{dataset}", split=split)
            raw_items = list(loader.load())
            if args.limit:
                raw_items = raw_items[: args.limit]
            logger.info("[%s/%s] %d raw items", dataset, split, len(raw_items))

            out_dir = Path(args.processed_dir) / args.reasoning_path_type / dataset / split
            out_dir.mkdir(parents=True, exist_ok=True)
            # Append-only shard handles (one file per subset_graph pair).
            shard_handles: Dict[str, any] = {}

            def _get_handle(key: str):
                if key not in shard_handles:
                    p = out_dir / key
                    shard_handles[key] = p.open("w", encoding="utf-8")
                    logger.info("opened shard %s", p)
                return shard_handles[key]

            try:
                for gt_str in args.graph_types:
                    gt = GraphType(gt_str)
                    print(f"\n=== {dataset}/{split} | graph_type = {gt.value} " + "=" * 20)
                    for idx, raw in enumerate(raw_items):
                        oid = str(raw.get("original_id", f"item_{idx}"))
                        t0 = time.perf_counter()
                        ep_dict, rho_or_reason = _process_one(
                            raw_item=raw, dataset=dataset, split=split,
                            target_graph_type=gt, seed=args.seed,
                            episode_builder=episode_builder,
                            rollout_engine=engine,
                            subset_assigner=subset_assigner,
                            serializer=serializer,
                        )
                        dt = time.perf_counter() - t0
                        if ep_dict is None:
                            total_fail += 1
                            print(f"  [{idx+1}/{len(raw_items)}] {oid[:30]:30s} FAIL: {rho_or_reason}")
                            continue

                        subset = ep_dict["rho_subset"].lower()
                        shard_key = f"{subset}_{gt.value.lower()}.jsonl"
                        f = _get_handle(shard_key)
                        f.write(json.dumps(ep_dict, ensure_ascii=False))
                        f.write("\n")
                        f.flush()
                        shard_counts[shard_key] += 1
                        total_rhos[gt.value].append(rho_or_reason)
                        total_ok += 1
                        print(f"  [{idx+1}/{len(raw_items)}] {oid[:30]:30s} "
                              f"rho={rho_or_reason:.3f} subset={ep_dict['rho_subset']} "
                              f"({dt:.1f}s)")
            finally:
                for h in shard_handles.values():
                    h.close()

    dt_total = time.perf_counter() - t_run_start

    print(f"\n=== SUMMARY ({llm.name}) {'=' * 40}")
    print(f"  ok={total_ok}  fail={total_fail}  wall={dt_total:.1f}s")
    for gt_name, rhos in total_rhos.items():
        if rhos:
            print(f"  {gt_name:20s} n={len(rhos)} mean_rho={sum(rhos)/len(rhos):.3f}")
    print(f"  shards written:")
    for k, v in sorted(shard_counts.items()):
        print(f"    {k}  ({v})")
    return 0 if total_fail == 0 else 2


# ============================================================
# CLI
# ============================================================


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoScope dataset construction pipeline")

    # data
    p.add_argument("--datasets", nargs="+", required=True,
                   choices=["musique", "2wikimhqa", "hotpotqa", "gsm8k", "math"])
    p.add_argument("--splits", nargs="+", default=["dev"],
                   choices=["train", "dev", "test"])
    p.add_argument("--graph-types", nargs="+",
                   default=["LINEAR", "FORK", "FORK_MERGE", "INDEPENDENT", "POLICY_ISOLATED"])
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on raw items per split.")

    # paths
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--reasoning-path-type", default="got", choices=["got", "cot", "tot"])

    # backends
    p.add_argument("--llm", default="template", choices=["template", "dashscope"])
    p.add_argument("--llm-model", default="qwen-plus")
    p.add_argument("--embedder", default="none", choices=["none", "dashscope"])
    p.add_argument("--embedder-model", default="text-embedding-v3")
    p.add_argument("--embedder-dim", type=int, default=1024)

    # rollout knobs
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=512)

    # run control
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--test", action="store_true",
                   help="Smoke-test mode: route output to tests/tmp/ instead of "
                        "data/processed/. Use for template LLM / dev runs.")

    args = p.parse_args(argv)
    if args.test:
        args.processed_dir = "tests/tmp/processed"
    return args


def main(argv: List[str] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    load_dotenv()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return run_build(args)


if __name__ == "__main__":
    sys.exit(main())
