"""
Full construction entry point.

Iterates over (dataset, split, graph_type) combinations and writes JSONL
shards to `data/processed/`. Intended for large-scale builds; use
`DatasetPipeline` directly for single-slice runs.

Usage:
    python -m coscope.data.scripts.build_all \\
        --datasets musique 2wikimhqa hotpotqa gsm8k math \\
        --splits dev test \\
        --data-dir data/raw --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from coscope.data.core.types import GraphType
from coscope.data.got.graph_templates import available_graph_types
from coscope.data.pipeline.dataset_pipeline import DatasetPipeline


# v2.1 default graph-type mixes: chosen so that every dataset/split yields
# S1, S2, and S3 episodes under the empirical rho distribution.
#   S1 source: hop_2 FORK (ρ ≈ 0.333 > 0.25)
#   S2 source: hop_2 LINEAR / POLICY_ISOLATED + hop_3/4 LINEAR / FORK / FORK_MERGE
#              (ρ ∈ (0.10, 0.25])
#   S3 source: INDEPENDENT (any hop) + 2-hop FORK_MERGE (ρ ≤ 0.10)
#   S4 source: POLICY_ISOLATED (auto-added for test splits unless disabled)
DEFAULT_GRAPH_TYPES_BY_DATASET = {
    "musique":         [GraphType.LINEAR, GraphType.FORK, GraphType.FORK_MERGE, GraphType.INDEPENDENT],
    "2wikimultihopqa": [GraphType.LINEAR, GraphType.FORK, GraphType.FORK_MERGE, GraphType.INDEPENDENT],
    "hotpotqa":        [GraphType.LINEAR, GraphType.FORK, GraphType.FORK_MERGE, GraphType.INDEPENDENT],
    "gsm8k":           [GraphType.LINEAR, GraphType.FORK, GraphType.FORK_MERGE, GraphType.INDEPENDENT],
    "math":            [GraphType.LINEAR, GraphType.FORK, GraphType.FORK_MERGE, GraphType.INDEPENDENT],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Full dataset construction")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--reasoning-path-type", default="got",
                        choices=["got", "cot", "tot"],
                        help="Subdirectory under processed/ (default: got)")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-s4", action="store_true", default=True,
                        help="Construct POLICY_ISOLATED for test splits (enabled by default)")
    parser.add_argument("--no-include-s4", dest="include_s4", action="store_false",
                        help="Disable automatic POLICY_ISOLATED injection for test splits")
    parser.add_argument("--no-enforce-coverage", dest="enforce_coverage", action="store_false",
                        default=True, help="Disable subset-coverage assertion after build")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    pipeline = DatasetPipeline(
        processed_dir=args.processed_dir,
        data_dir=args.data_dir,
        reasoning_path_type=args.reasoning_path_type,
    )

    for dataset in args.datasets:
        base_types: List[GraphType] = list(
            DEFAULT_GRAPH_TYPES_BY_DATASET.get(dataset, [GraphType.LINEAR])
        )
        for split in args.splits:
            target_types = list(base_types)
            if args.include_s4 and split == "test":
                target_types.append(GraphType.POLICY_ISOLATED)
            print(f"=> {dataset}/{split}  graph_types={[t.value for t in target_types]}")
            written = pipeline.run(
                dataset=dataset,
                split=split,
                target_graph_types=target_types,
                max_workers=args.max_workers,
                limit=args.limit,
                enforce_coverage=args.enforce_coverage,
            )
            for path, count in written.items():
                print(f"   {path}: {count}")


if __name__ == "__main__":
    main()
