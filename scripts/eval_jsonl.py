"""
CLI entry point for Mode-B JSONL evaluation (§14.5.1 algorithm evaluation).

Usage
-----
    python -m scripts.eval_jsonl \\
        --shards tests/tmp/processed/got/musique/dev/*.jsonl \\
        --variants a1 a2 a3 a4 a4_nofb a4_norerank \\
        --k 10 \\
        --output tests/tmp/processed/got/musique/dev/eval_report.json

Reads each JSONL shard, replays the retrieval requests through a fresh
``CoScope`` engine per episode (memories preloaded verbatim so memory_id
references match ground_truth), runs each requested variant, and prints
stratified S1/S2/S3/S4 tables plus an optional JSON dump for later plotting.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import List

from engine import CoScope, RandomEmbeddingProvider
from evaluation.jsonl_runner import (
    DEFAULT_VARIANTS,
    evaluate_jsonl,
    format_stratified_table,
)


def _make_engine_factory(
    embedder: str,
    dim: int,
    st_model: str,
    dashscope_model: str = "text-embedding-v3",
    dashscope_dim: int = 1024,
    cache_path: str = None,
):
    """Return a zero-arg factory creating a fresh CoScope engine.

    - 'random' : deterministic hash-based embeddings (no API cost, smoke).
    - 'st'     : sentence-transformers offline embedder (no API cost, real
                 semantic signal; first use downloads model weights).
    - 'default': let CoScope pick from config.yaml (may need API keys).

    A fresh engine is used per episode so memory and scope state stay isolated.
    """
    embedder = embedder.lower()
    if embedder == "random":
        def _factory_random():
            return CoScope(embedding_provider=RandomEmbeddingProvider(dimension=dim))
        return _factory_random
    if embedder == "st":
        # Construct the embedder once; reuse across episodes to avoid
        # re-downloading / re-loading the model per episode.
        from embedding import SentenceTransformerEmbedder
        shared_embedder = SentenceTransformerEmbedder(model_name=st_model)
        def _factory_st():
            return CoScope(embedding_provider=shared_embedder)
        return _factory_st
    if embedder == "default":
        def _factory_default():
            return CoScope()
        return _factory_default
    if embedder == "dashscope":
        # Real DashScope text-embedding-v3 (paper setup). Wrap in disk
        # cache so re-runs / variant sweeps don't re-pay API cost.
        from embedding import DashScopeEmbedder, DiskCachedEmbedder
        from pathlib import Path as _Path
        inner = DashScopeEmbedder(model=dashscope_model, dim=dashscope_dim)
        if cache_path:
            shared_embedder = DiskCachedEmbedder(
                inner, _Path(cache_path)
            )
        else:
            shared_embedder = inner
        def _factory_dashscope():
            return CoScope(embedding_provider=shared_embedder)
        return _factory_dashscope
    raise ValueError(
        f"Unknown embedder '{embedder}'; expected one of 'random', 'st', 'dashscope', 'default'."
    )


def _expand_shards(patterns: List[str]) -> List[Path]:
    """Expand glob patterns and return unique paths in a stable order."""
    seen: List[Path] = []
    found = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for m in matches:
            p = Path(m).resolve()
            if p not in found:
                found.add(p)
                seen.append(p)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mode-B JSONL evaluation (§14.5.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        required=True,
        help="JSONL shard paths (glob patterns allowed).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Pipeline variants to evaluate.",
    )
    parser.add_argument("--k", type=int, default=10, help="Recall/MRR cutoff.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "recall_at_k",
            "evidence_hit_rate",
            "mrr_at_k",
            "fallback_rate",
            "false_merge_rate",
            "content_false_merge_rate",
            "first_stage_savings",
        ],
        help="Metrics to print as stratified tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the full report (includes per-episode rows).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per evaluated episode.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level for the coscope engine (INFO is noisy).",
    )
    parser.add_argument(
        "--embedder",
        choices=["random", "st", "dashscope", "default"],
        default="random",
        help="'random' deterministic hash (smoke, zero setup); "
             "'st' offline sentence-transformers (no API); "
             "'dashscope' real Qwen text-embedding-v3 (paper setup, needs "
             "DASHSCOPE_API_KEY; recommend pairing with --cache-dir); "
             "'default' follows config.yaml.",
    )
    parser.add_argument(
        "--dashscope-model",
        default="text-embedding-v3",
        help="DashScope embedding model (paper main: text-embedding-v3).",
    )
    parser.add_argument(
        "--dashscope-dim",
        type=int,
        default=1024,
        help="DashScope embedding dim (v3 supports 512/768/1024).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="If set, use a disk-cached wrapper around the embedder. "
             "The cache is content-addressed (sha1 of model|dim|text), "
             "so it is safe across runs and variants. Strongly recommended "
             "with --embedder dashscope.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension used by the random provider (ignored by 'st').",
    )
    parser.add_argument(
        "--st-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model identifier (used only by --embedder=st).",
    )
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging.basicConfig(level=level)
    # Engine and pipeline submodules configure their own handlers; align them.
    for name in ("coscope", "engine", "retrieval", "coscope.memory"):
        logging.getLogger(name).setLevel(level)

    shard_paths = _expand_shards(args.shards)
    if not shard_paths:
        print("No shards matched the given patterns", file=sys.stderr)
        return 2
    print(f"Loaded {len(shard_paths)} shard(s):")
    for p in shard_paths:
        print(f"  - {p}")

    cache_path = None
    if args.cache_dir:
        from pathlib import Path as _P
        suffix = (
            f"{args.dashscope_model.replace('/', '_')}-d{args.dashscope_dim}.sqlite"
            if args.embedder == "dashscope"
            else f"{args.st_model.replace('/', '_')}.sqlite"
            if args.embedder == "st"
            else "random.sqlite"
        )
        cache_path = str(_P(args.cache_dir) / suffix)
        _P(args.cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"Embedder cache: {cache_path}")

    engine_factory = _make_engine_factory(
        args.embedder,
        args.embedding_dim,
        args.st_model,
        dashscope_model=args.dashscope_model,
        dashscope_dim=args.dashscope_dim,
        cache_path=cache_path,
    )

    episode_runs, stratified = evaluate_jsonl(
        shard_paths=shard_paths,
        variants=args.variants,
        k=args.k,
        engine_factory=engine_factory,
        verbose=args.verbose,
    )
    print(f"\nEvaluated {len(episode_runs)} episode(s)\n")

    for metric in args.metrics:
        print(format_stratified_table(stratified, metric=metric))
        print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stratified": stratified.to_dict(),
            "episodes": [er.to_dict() for er in episode_runs],
            "args": {
                "shards": [str(p) for p in shard_paths],
                "variants": list(args.variants),
                "k": args.k,
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Full report written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
