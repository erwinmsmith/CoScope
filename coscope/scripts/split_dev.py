"""
Generate and freeze dev splits for GSM8K / MATH from their train sets.

Usage:
    python -m coscope.scripts.split_dev --dataset gsm8k --data-dir data/raw/gsm8k
    python -m coscope.scripts.split_dev --dataset math --data-dir data/raw/math
"""

from __future__ import annotations

import argparse

from coscope.utils.loaders import get_loader
from coscope.utils.split.split_manager import SplitManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze dev-id lists for GSM8K / MATH.")
    parser.add_argument("--dataset", required=True, choices=["gsm8k", "math"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loader = get_loader(args.dataset, data_dir=args.data_dir, split="train")
    train_items = loader.load()
    manager = SplitManager(interim_dir=args.interim_dir)

    if args.dataset == "math":
        dev, remaining = manager.generate_math_dev(train_items, dev_ratio=args.ratio, seed=args.seed)
    else:
        dev, remaining = manager.generate_uniform_dev(train_items, dev_ratio=args.ratio, seed=args.seed)

    dev_ids = [item["original_id"] for item in dev]
    path = manager.freeze_dev_ids(args.dataset, dev_ids)
    # Register remaining as train for leak checks.
    manager.register(args.dataset, "train", [item["original_id"] for item in remaining])
    print(f"{args.dataset}: {len(dev)} dev / {len(remaining)} train -> {path}")


if __name__ == "__main__":
    main()
