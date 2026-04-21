"""
Statistics reporter.

Generates the `{split}_stats.json` report described in spec §14.2.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from coscope.data.core.types import Episode, SubsetLabel


SCHEMA_VERSION = "2.0"


class SubsetCoverageError(RuntimeError):
    """Raised when a dataset split fails the S1/S2/S3(/S4) coverage check."""


class StatsReporter:
    """Aggregate statistics for a set of episodes."""

    def build(self, episodes: Iterable[Episode], *, dataset: str, split: str) -> Dict[str, Any]:
        episodes = list(episodes)
        rho_by_subset: Dict[str, List[float]] = defaultdict(list)
        subset_counts: Counter = Counter()
        graph_type_counts: Counter = Counter()
        hop_counts: Counter = Counter()
        gold_coverage_hits = 0
        gold_coverage_total = 0
        s4_count = 0

        for ep in episodes:
            subset_counts[ep.rho_subset.value] += 1
            rho_by_subset[ep.rho_subset.value].append(ep.rho)
            graph_type_counts[self._graph_key(ep)] += 1
            hop_counts[ep.hop_count] += 1
            if ep.s4_eligible:
                s4_count += 1
            gold_entries = [
                e for e in ep.memory_entries
                if (e.metadata or {}).get("is_gold_evidence") is True
            ]
            gold_coverage_total += 1
            if gold_entries:
                gold_coverage_hits += 1

        rho_stats: Dict[str, Dict[str, float]] = {}
        for label in (SubsetLabel.S1, SubsetLabel.S2, SubsetLabel.S3):
            values = rho_by_subset.get(label.value, [])
            if not values:
                continue
            rho_stats[label.value] = {
                "mean": round(statistics.fmean(values), 4),
                "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "n": len(values),
            }

        return {
            "dataset": dataset,
            "split": split,
            "total_episodes": len(episodes),
            "subset_counts": dict(subset_counts),
            "s4_count": s4_count,
            "graph_type_counts": dict(graph_type_counts),
            "rho_stats": rho_stats,
            "hop_count_distribution": {str(k): v for k, v in sorted(hop_counts.items())},
            "gold_evidence_coverage": (
                round(gold_coverage_hits / gold_coverage_total, 4)
                if gold_coverage_total
                else 0.0
            ),
            "schema_version": SCHEMA_VERSION,
        }

    def write(
        self,
        episodes: Iterable[Episode],
        *,
        path: Union[str, Path],
        dataset: str,
        split: str,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self.build(episodes, dataset=dataset, split=split)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path

    # ------------------------------------------------------------------

    @staticmethod
    def _graph_key(ep: Episode) -> str:
        gt = ep.graph_type
        if hasattr(gt, "value"):
            return gt.value
        return str(gt)

    # ------------------------------------------------------------------

    @staticmethod
    def check_coverage(
        episodes: Iterable[Episode],
        *,
        split: str,
        require_s4: bool = False,
        dataset: str = "",
    ) -> Dict[str, int]:
        """
        Enforce per-dataset S1/S2/S3 coverage and (for test splits) S4 coverage.

        Rule (per v2.1 §16.2):
          - train/dev/test: S1, S2, S3 must each have >= 1 episode.
          - test (or when require_s4=True): S4 must also have >= 1 episode.

        Raises `SubsetCoverageError` on failure. Returns the per-subset counts.
        """
        eps = list(episodes)
        counts = {label: 0 for label in ("S1", "S2", "S3", "S4")}
        for ep in eps:
            counts[ep.rho_subset.value] += 1
            if ep.s4_eligible:
                counts["S4"] += 1

        required = ["S1", "S2", "S3"]
        if require_s4 or split == "test":
            required.append("S4")

        missing = [s for s in required if counts[s] == 0]
        if missing:
            raise SubsetCoverageError(
                f"Dataset '{dataset}' split '{split}' is missing subsets: "
                f"{missing}. Got counts {counts}. "
                f"Adjust `target_graph_types` to include combinations that yield "
                f"the missing labels (see v2.1 §9.1 ρ distribution table)."
            )
        return counts
