"""
S4 False-Merge-Rate checker.

See §13.5. A bucket is considered a "false merge" when it contains agents with
mixed clearance levels (a Verifier clearance=2 agent sharing a bucket with
clearance=1 Planner/Solver).
"""

from __future__ import annotations

from typing import List, Sequence

from core.types import Agent
from core.types import Episode


def check_false_merge(bucket: Sequence[Agent]) -> bool:
    """True iff the bucket exhibits a clearance-level mix (policy conflict)."""
    clearances = {int(a.config.policy.max_clearance or 0) for a in bucket}
    return len(clearances) > 1


def compute_fmr(
    episodes: List[Episode],
    buckets_per_episode: List[List[Sequence[Agent]]],
) -> float:
    """
    FMR = (# S4 episodes containing at least one false-merge bucket) / (# S4 episodes).
    Returns 0.0 when there are no S4 episodes.
    """
    s4_indices = [i for i, ep in enumerate(episodes) if ep.s4_eligible]
    if not s4_indices:
        return 0.0
    bad = 0
    for i in s4_indices:
        buckets = buckets_per_episode[i] if i < len(buckets_per_episode) else []
        if any(check_false_merge(bucket) for bucket in buckets):
            bad += 1
    return bad / len(s4_indices)
