"""
Subset assigner (S1 / S2 / S3 / S4).

See `data/docs/GoT_Dataset_Requirements_v2.md` section 10.

- S1 / S2 / S3 are mutually exclusive and determined by rho thresholds.
- S4 is orthogonal: an episode whose `graph_type == POLICY_ISOLATED` is marked
  `s4_eligible=True` and `policy_conflict=True`. It keeps its S1/S2/S3 label in
  `rho_subset` but additionally qualifies for the test-only S4 shard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from coscope.data.core.types import GoTGraph, GraphType, SubsetLabel


@dataclass
class SubsetThresholds:
    """
    Cutoffs for rho-based subset labels (v2.1 empirical tuning).

        S1: rho >  s1_min
        S2: s2_min  <  rho  <=  s1_min
        S3: rho <=  s2_min

    Defaults match `data/config/subset_thresholds.yaml`.
    """

    s1_min: float = 0.25
    s2_min: float = 0.10

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SubsetThresholds":
        import yaml  # local import so the base package has no hard dep
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        s1 = float(cfg.get("s1_threshold", cls.s1_min))
        # Accept both `s2_lower` and `s3_upper` (they must be equal).
        s2 = cfg.get("s2_lower", cfg.get("s3_upper", cls.s2_min))
        s2 = float(s2)
        if "s2_lower" in cfg and "s3_upper" in cfg:
            if float(cfg["s2_lower"]) != float(cfg["s3_upper"]):
                raise ValueError(
                    "subset_thresholds.yaml: s2_lower must equal s3_upper "
                    f"(got s2_lower={cfg['s2_lower']}, s3_upper={cfg['s3_upper']})"
                )
        if not (0.0 <= s2 <= s1 <= 1.0):
            raise ValueError(
                f"invalid subset thresholds: require 0 <= s2_min({s2}) <= s1_min({s1}) <= 1"
            )
        return cls(s1_min=s1, s2_min=s2)


DEFAULT_THRESHOLDS = SubsetThresholds()


def load_default_thresholds() -> SubsetThresholds:
    """
    Resolve thresholds from `data/config/subset_thresholds.yaml` when present,
    otherwise return `DEFAULT_THRESHOLDS`.
    """
    candidate = Path("data/config/subset_thresholds.yaml")
    if candidate.exists():
        try:
            return SubsetThresholds.from_yaml(candidate)
        except Exception:
            # fall back to hard-coded defaults silently; callers can log
            return DEFAULT_THRESHOLDS
    return DEFAULT_THRESHOLDS


@dataclass
class SubsetAssignment:
    rho_subset: SubsetLabel
    policy_conflict: bool
    s4_eligible: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "rho_subset": self.rho_subset.value,
            "policy_conflict": self.policy_conflict,
            "s4_eligible": self.s4_eligible,
        }


class SubsetAssigner:
    def __init__(self, thresholds: Optional[SubsetThresholds] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def assign(self, rho: float, got_graph: GoTGraph) -> SubsetAssignment:
        rho_subset = self._rho_label(rho)
        policy_conflict = self._is_policy_isolated(got_graph)
        return SubsetAssignment(
            rho_subset=rho_subset,
            policy_conflict=policy_conflict,
            s4_eligible=policy_conflict,
        )

    # ------------------------------------------------------------------

    def _rho_label(self, rho: float) -> SubsetLabel:
        t = self.thresholds
        if rho > t.s1_min:
            return SubsetLabel.S1
        if rho > t.s2_min:
            return SubsetLabel.S2
        return SubsetLabel.S3

    @staticmethod
    def _is_policy_isolated(got_graph: GoTGraph) -> bool:
        graph_type = got_graph.graph_type
        if isinstance(graph_type, GraphType):
            return graph_type == GraphType.POLICY_ISOLATED
        return str(graph_type) == GraphType.POLICY_ISOLATED.value
