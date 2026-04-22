"""
Cross-episode / cross-file validators.

Single-episode validation lives in `coscope.data.auth.policy_validator`.
This module adds batch-level checks (leakage, S4 placement, distribution).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

from coscope.data.core.types import Episode, SubsetLabel
from coscope.data.split.subset_assigner import (
    SubsetThresholds,
    load_default_thresholds,
)


@dataclass
class BatchValidationResult:
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class BatchValidator:
    """Check invariants across a collection of episodes."""

    def __init__(self, thresholds: "SubsetThresholds | None" = None):
        self.thresholds = thresholds or load_default_thresholds()

    def validate(self, episodes: Iterable[Episode]) -> BatchValidationResult:
        result = BatchValidationResult()
        episodes = list(episodes)
        if not episodes:
            result.add_warning("empty episode batch")
            return result

        self._check_no_leakage(episodes, result)
        self._check_s4_placement(episodes, result)
        self._check_subset_rho_consistency(episodes, result)
        self._populate_stats(episodes, result)
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _check_no_leakage(episodes: List[Episode], result: BatchValidationResult) -> None:
        split_to_ids: Dict[str, Set[str]] = defaultdict(set)
        for ep in episodes:
            split_to_ids[ep.split].add(ep.original_id)
        train = split_to_ids.get("train", set())
        dev = split_to_ids.get("dev", set())
        test = split_to_ids.get("test", set())
        if train & test:
            result.add_error(f"original_id leakage between train/test: {len(train & test)} ids")
        if dev & test:
            result.add_error(f"original_id leakage between dev/test: {len(dev & test)} ids")
        if train & dev:
            result.add_warning(
                f"original_id overlap between train/dev: {len(train & dev)} ids (acceptable only if intentional)"
            )

    @staticmethod
    def _check_s4_placement(episodes: List[Episode], result: BatchValidationResult) -> None:
        for ep in episodes:
            if ep.s4_eligible and ep.split != "test":
                result.add_error(
                    f"S4-eligible episode {ep.episode_id} placed in split={ep.split}, "
                    f"must be test"
                )

    def _check_subset_rho_consistency(
        self, episodes: List[Episode], result: BatchValidationResult
    ) -> None:
        s1_min = self.thresholds.s1_min
        s2_min = self.thresholds.s2_min
        for ep in episodes:
            rho = ep.rho
            expected: SubsetLabel
            if rho > s1_min:
                expected = SubsetLabel.S1
            elif rho > s2_min:
                expected = SubsetLabel.S2
            else:
                expected = SubsetLabel.S3
            if ep.rho_subset != expected:
                result.add_warning(
                    f"episode {ep.episode_id}: rho={rho} but rho_subset={ep.rho_subset.value} "
                    f"(expected {expected.value} under thresholds s1>{s1_min}, s2>{s2_min})"
                )

    @staticmethod
    def _populate_stats(episodes: List[Episode], result: BatchValidationResult) -> None:
        result.stats["total"] = len(episodes)
        result.stats["s4_eligible"] = sum(1 for ep in episodes if ep.s4_eligible)
        for label in SubsetLabel:
            result.stats[f"rho_{label.value}"] = sum(
                1 for ep in episodes if ep.rho_subset == label
            )
