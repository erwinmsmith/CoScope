"""
Rebuild ONLY the POLICY_ISOLATED (S4) shard for a (rpt, dataset, split).

Why this exists
---------------
``scripts/run_stage_a_rpt.sh`` performs directory-level idempotency: if the
shard directory exists and is non-empty, it skips the build entirely. That is
fine for fresh rpts but blocks the workflow where we keep S1-S3 intact and
only want to refresh S4 after regenerating the restricted-evidence interim
file (``data/interim/restricted/{dataset}_restricted.jsonl``).

``Serializer.write_jsonl`` opens in ``"w"`` mode, so calling
``DatasetPipeline.run`` with ``target_graph_types=[GraphType.POLICY_ISOLATED]``
overwrites only ``s4_policy_isolated.jsonl`` and leaves S1-S3 shards
untouched.

Pre/post original_id alignment check
------------------------------------
Before deleting / overwriting the existing S4 shard we capture the set of
``original_id`` values from the current shard and compare it to the freshly
built one. The two sets must be equal (membership; order may differ); a diff
indicates the loader, restricted interim, or graph templates have drifted and
downstream eval reports may no longer be comparable.

Usage
-----
    python -m scripts.rebuild_s4_only \
        --rpt got --dataset 2wikimhqa --split test
    python -m scripts.rebuild_s4_only \
        --rpt tot --dataset hotpotqa --split test --skip-alignment-check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Set

from core.types import GraphType
from construction.dataset_pipeline import DatasetPipeline


def _read_original_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            oid = row.get("original_id")
            if oid is not None:
                ids.add(str(oid))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rpt", required=True, choices=["got", "cot", "tot"],
                        help="Reasoning path type (subdirectory under processed/).")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (musique | 2wikimhqa | hotpotqa | gsm8k | math).")
    parser.add_argument("--split", default="test",
                        help="Split (S4 is only generated for test by build_all).")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-alignment-check", action="store_true",
                        help="Do not enforce equality of original_id sets pre vs post rebuild "
                             "(use for first-time S4 build where no prior shard exists).")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    shard_dir = Path(args.processed_dir) / args.rpt / args.dataset / args.split
    s4_path = shard_dir / "s4_policy_isolated.jsonl"

    pre_ids = _read_original_ids(s4_path)
    print(f"[pre]  {s4_path} -> {len(pre_ids)} original_ids")

    pipeline = DatasetPipeline(
        processed_dir=args.processed_dir,
        data_dir=args.data_dir,
        reasoning_path_type=args.rpt,
    )

    print(f"[build] rebuilding S4 only: rpt={args.rpt} dataset={args.dataset} split={args.split}")
    written = pipeline.run(
        dataset=args.dataset,
        split=args.split,
        target_graph_types=[GraphType.POLICY_ISOLATED],
        max_workers=args.max_workers,
        limit=args.limit,
        enforce_coverage=False,
    )
    for path, count in written.items():
        print(f"   {path}: {count}")

    post_ids = _read_original_ids(s4_path)
    print(f"[post] {s4_path} -> {len(post_ids)} original_ids")

    if args.skip_alignment_check:
        print("[align] skipped (--skip-alignment-check)")
        return 0

    if not pre_ids:
        print("[align] no prior shard; nothing to compare. Re-run with another rpt to cross-check.")
        return 0

    only_pre = pre_ids - post_ids
    only_post = post_ids - pre_ids
    if not only_pre and not only_post:
        print(f"[align] OK: original_id sets identical ({len(pre_ids)} ids).")
        return 0

    print(f"[align] MISMATCH: only_in_pre={len(only_pre)}  only_in_post={len(only_post)}")
    sample_pre = list(only_pre)[:5]
    sample_post = list(only_post)[:5]
    if sample_pre:
        print(f"   sample only_in_pre:  {sample_pre}")
    if sample_post:
        print(f"   sample only_in_post: {sample_post}")
    print("   downstream eval reports referencing the prior S4 shard may not be comparable.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
