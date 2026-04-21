"""
Pre-compute and cache rho values for all processed episodes of a dataset.

This script is optional: rho is also computed inline inside EpisodeBuilder.
Run it when you want to warm the on-disk cache before large-scale construction,
or to re-verify previously generated episodes.

Usage:
    python -m coscope.data.scripts.compute_rho --dataset musique --split dev
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coscope.data.got.rho_calculator import RhoCalculator
from coscope.data.output.serializer import Serializer


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm the rho cache for serialized episodes.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--cache-dir", default="data/interim/rho_cache")
    args = parser.parse_args()

    root = Path(args.processed_dir) / args.dataset / args.split
    if not root.exists():
        raise SystemExit(f"no processed episodes at {root}")

    serializer = Serializer()
    calc = RhoCalculator(cache_dir=args.cache_dir)

    updated = 0
    for file in sorted(root.glob("*.jsonl")):
        for ep in serializer.read_jsonl(file):
            calc.compute(
                got_graph=ep.got_graph,
                memory_entries=ep.memory_entries,
                dataset=args.dataset,
                episode_id=ep.episode_id,
                use_cache=True,
            )
            updated += 1
    calc.flush()
    print(f"Updated cache for {updated} episodes at {Path(args.cache_dir) / f'{args.dataset}_rho_cache.json'}")


if __name__ == "__main__":
    main()
