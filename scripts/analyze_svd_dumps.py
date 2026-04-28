"""Offline analysis of per-bucket SVD artifacts produced by
``--dump-svd-artifacts`` in ``scripts/eval_jsonl.py``.

For each bucket dump (Q, S, Z, W_final, K_proj saved as .npz plus a small JSON
of bucket metadata) this script computes:

  * n_queries distribution per graph_type (bucket "thinness")
  * effective rank ``r_eff = #{i : S[i] / S[0] > eps}`` (default eps = 1e-3)
  * top-1 / top-3 spectral energy fraction Σ S[:r]^2 / Σ S^2
  * subspace overlap between query subspace (V[:r]) and the actual gold
    direction in the candidate pool (proxy: max cos(K_proj_i, Z_j))

The output is a JSON summary table per graph_type and a small markdown
table suitable to drop into ``docs/Experiments.md``.

Usage::

    python scripts/analyze_svd_dumps.py \\
        --dump-root data/processed/got/musique/test/svd_scan/dumps_r64 \\
        --eval-json data/processed/got/musique/test/svd_scan/svd_scan_r64.json \\
        --output    data/processed/got/musique/test/svd_scan/svd_scan_r64.summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, median
from typing import Dict, List

import numpy as np


def _classify_episode_id(episode_id: str) -> str:
    """Pull graph_type out of episode_id ``..._{rpt}_{graph_type}``."""
    for gt in ("FORK_MERGE", "POLICY_ISOLATED", "INDEPENDENT", "LINEAR", "FORK"):
        if episode_id.endswith("_" + gt):
            return gt
    return "UNKNOWN"


def _energy_frac(S: np.ndarray, top: int) -> float:
    if S.size == 0:
        return 0.0
    total = float(np.sum(S ** 2))
    if total <= 0:
        return 0.0
    return float(np.sum(S[:top] ** 2)) / total


def _effective_rank(S: np.ndarray, eps: float = 1e-3) -> int:
    if S.size == 0:
        return 0
    s0 = float(S[0]) if S[0] > 0 else 1.0
    return int(np.sum((S / s0) > eps))


def _max_query_candidate_cos(Z: np.ndarray, K_proj: np.ndarray) -> float:
    """Proxy for "is the gold candidate visible in the query subspace?"

    Returns the median over candidates of ``max_j cos(K_proj_i, Z_j)``.
    Higher = candidates align well with at least one query direction.
    """
    if Z.size == 0 or K_proj.size == 0:
        return float("nan")
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    Kn = K_proj / np.maximum(np.linalg.norm(K_proj, axis=1, keepdims=True), 1e-12)
    sims = Kn @ Zn.T  # (m, n)
    if sims.size == 0:
        return float("nan")
    per_cand_max = sims.max(axis=1)
    return float(np.median(per_cand_max))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", required=True,
                        help="Directory containing per-episode subdirs of .npz/.json dumps.")
    parser.add_argument("--eval-json", default=None,
                        help="Optional eval_jsonl output to join recall@10 onto each episode.")
    parser.add_argument("--output", default=None,
                        help="Optional JSON path for the structured summary.")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="Threshold for effective-rank counting (S[i]/S[0] > eps).")
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.dump_root, "*", "*.npz")))
    if not npz_files:
        print(f"No .npz files found under {args.dump_root}")
        return 2
    print(f"Loaded {len(npz_files)} bucket dumps from {len(set(os.path.dirname(f) for f in npz_files))} episodes")

    # Per-bucket records
    rows = []
    for npz_path in npz_files:
        json_path = npz_path[:-4] + ".json"
        if not os.path.exists(json_path):
            continue
        with np.load(npz_path) as d:
            S = d["S"]
            Z = d["Z"]
            K_proj = d["K_proj"]
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        episode_id = meta.get("episode_id", "_unknown")
        gt = _classify_episode_id(episode_id)
        rows.append({
            "episode_id": episode_id,
            "graph_type": gt,
            "bucket_id": meta.get("bucket_id"),
            "n_queries": int(meta.get("n_queries", 0)),
            "n_pool": int(meta.get("n_pool_candidates", 0)),
            "r_svd_config": int(meta.get("svd_rank_config", 0)),
            "r_actual": int(meta.get("r_svd", 0)),
            "r_effective": _effective_rank(S, eps=args.eps),
            "top1_energy": _energy_frac(S, 1),
            "top3_energy": _energy_frac(S, 3),
            "max_qc_cos_median": _max_query_candidate_cos(Z, K_proj),
        })

    # Optional join with eval recall
    recall_by_ep: Dict[str, float] = {}
    mrr_by_ep: Dict[str, float] = {}
    if args.eval_json and os.path.exists(args.eval_json):
        with open(args.eval_json, "r", encoding="utf-8") as f:
            ev = json.load(f)
        for ep in ev.get("episodes", []):
            ep_id = ep.get("episode_id")
            for vname, run in (ep.get("variants") or {}).items():
                if vname == "a5":
                    recall_by_ep[ep_id] = run.get("recall_at_k", float("nan"))
                    mrr_by_ep[ep_id] = run.get("mrr_at_k", float("nan"))

    # Aggregate per graph_type
    by_gt: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_gt[r["graph_type"]].append(r)

    summary = {}
    for gt, items in sorted(by_gt.items()):
        nqs = [r["n_queries"] for r in items]
        rs_eff = [r["r_effective"] for r in items]
        e1 = [r["top1_energy"] for r in items]
        e3 = [r["top3_energy"] for r in items]
        qc = [r["max_qc_cos_median"] for r in items if not np.isnan(r["max_qc_cos_median"])]
        eps_in_gt = {r["episode_id"] for r in items}
        recall = [recall_by_ep[e] for e in eps_in_gt if e in recall_by_ep]
        mrr = [mrr_by_ep[e] for e in eps_in_gt if e in mrr_by_ep]
        summary[gt] = {
            "n_buckets": len(items),
            "n_episodes": len(eps_in_gt),
            "n_queries_median": median(nqs),
            "n_queries_max": max(nqs),
            "r_effective_median": median(rs_eff),
            "r_effective_max": max(rs_eff),
            "top1_energy_mean": mean(e1) if e1 else 0.0,
            "top3_energy_mean": mean(e3) if e3 else 0.0,
            "max_qc_cos_median_mean": mean(qc) if qc else float("nan"),
            "a5_recall_mean_on_dumped": mean(recall) if recall else None,
            "a5_mrr_mean_on_dumped": mean(mrr) if mrr else None,
        }

    # Print markdown table
    print()
    print("### Per-bucket SVD diagnostics (graph_type level)\n")
    header = ("graph_type", "n_buckets", "n_q (med/max)",
              "r_eff (med/max)", "top1 E", "top3 E",
              "median max-cos(K,Z)", "a5 R@10", "a5 MRR@10")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for gt, s in summary.items():
        row = [
            gt,
            str(s["n_buckets"]),
            f"{s['n_queries_median']}/{s['n_queries_max']}",
            f"{s['r_effective_median']}/{s['r_effective_max']}",
            f"{s['top1_energy_mean']:.3f}",
            f"{s['top3_energy_mean']:.3f}",
            f"{s['max_qc_cos_median_mean']:.3f}",
            f"{s['a5_recall_mean_on_dumped']:.4f}" if s['a5_recall_mean_on_dumped'] is not None else "-",
            f"{s['a5_mrr_mean_on_dumped']:.4f}" if s['a5_mrr_mean_on_dumped'] is not None else "-",
        ]
        print("| " + " | ".join(row) + " |")
    print()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"per_graph_type": summary, "per_bucket": rows}, f,
                      ensure_ascii=False, indent=2)
        print(f"Summary written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
