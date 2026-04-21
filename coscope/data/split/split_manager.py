"""
Split manager.

Responsibilities:
- Maintain the canonical train/dev/test membership for each dataset, especially
  for GSM8K / MATH where no official dev split exists and we sample from train.
- Freeze dev ids to disk (`data/interim/{dataset}_dev_ids.txt`) after the first
  sampling so later runs are reproducible.
- Provide leak detection: train and test `original_id` sets must be disjoint.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


class SplitManager:
    """In-memory registry + disk-backed freeze for (dataset, split) membership."""

    def __init__(self, interim_dir: Union[str, Path] = "data/interim"):
        self.interim_dir = Path(interim_dir)
        # dataset -> split -> Set[original_id]
        self._membership: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    # ------------------------------------------------------------------
    # Membership queries
    # ------------------------------------------------------------------

    def register(self, dataset: str, split: str, original_ids: Iterable[str]) -> None:
        """Register a batch of original_ids for a (dataset, split) pair."""
        self._membership[dataset][split].update(str(x) for x in original_ids)

    def get_split(self, dataset: str, original_id: str) -> Optional[str]:
        """Return the split name containing `original_id`, or None."""
        splits = self._membership.get(dataset, {})
        for split_name, ids in splits.items():
            if original_id in ids:
                return split_name
        return None

    def contains(self, dataset: str, split: str, original_id: str) -> bool:
        return original_id in self._membership.get(dataset, {}).get(split, set())

    # ------------------------------------------------------------------
    # GSM8K / MATH dev sampling
    # ------------------------------------------------------------------

    def generate_math_dev(
        self,
        train_items: List[Dict[str, Any]],
        dev_ratio: float = 0.10,
        seed: int = 42,
        category_key: str = "math_category",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Stratified dev sampling over `math_category`. Returns `(dev, remaining_train)`.
        """
        rng = random.Random(seed)
        by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in train_items:
            cat = item.get(category_key) or "unknown"
            by_category[cat].append(item)
        dev: List[Dict[str, Any]] = []
        remaining: List[Dict[str, Any]] = []
        for cat, items in by_category.items():
            rng.shuffle(items)
            n_dev = max(1, int(len(items) * dev_ratio))
            dev.extend(items[:n_dev])
            remaining.extend(items[n_dev:])
        return dev, remaining

    def generate_uniform_dev(
        self,
        train_items: List[Dict[str, Any]],
        dev_ratio: float = 0.10,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Unstratified dev sampling. Used by GSM8K where no category exists."""
        rng = random.Random(seed)
        items = list(train_items)
        rng.shuffle(items)
        n_dev = max(1, int(len(items) * dev_ratio))
        return items[:n_dev], items[n_dev:]

    # ------------------------------------------------------------------
    # Dev-id freeze I/O
    # ------------------------------------------------------------------

    def dev_ids_path(self, dataset: str) -> Path:
        return self.interim_dir / f"{dataset}_dev_ids.txt"

    def freeze_dev_ids(self, dataset: str, ids: Iterable[str]) -> Path:
        path = self.dev_ids_path(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        sorted_ids = sorted({str(x) for x in ids})
        with open(path, "w", encoding="utf-8") as f:
            for _id in sorted_ids:
                f.write(f"{_id}\n")
        # Also register in memory.
        self.register(dataset, "dev", sorted_ids)
        return path

    def load_dev_ids(self, dataset: str) -> Set[str]:
        path = self.dev_ids_path(dataset)
        if not path.exists():
            return set()
        with open(path, "r", encoding="utf-8") as f:
            ids = {line.strip() for line in f if line.strip()}
        self.register(dataset, "dev", ids)
        return ids

    # ------------------------------------------------------------------
    # Leak detection
    # ------------------------------------------------------------------

    def check_no_leakage(
        self, dataset: str, left: str = "train", right: str = "test"
    ) -> bool:
        """True iff no `original_id` is shared between `left` and `right` splits."""
        left_ids = self._membership.get(dataset, {}).get(left, set())
        right_ids = self._membership.get(dataset, {}).get(right, set())
        return len(left_ids & right_ids) == 0
