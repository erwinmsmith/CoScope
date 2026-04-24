"""
ToT rho calculator.

Per `coscope/data/docs/GoT_Dataset_Requirements_v2.md` §18.5: ρ for ToT is "对应 GoT
FORK 逻辑 ... 各分支路径 hop-workspace 完全不重叠，只有 Planner 的初始
task-shared 是公共的，预期 ρ ≤ 0.10 (S3 区间)". The numerical ρ formula is
identical to GoT's; the FORK graph template already encodes the branch
independence, so `RhoCalculator` from `coscope.data.got.rho_calculator` is
re-exported as the ToT calculator.
"""

from __future__ import annotations

from coscope.graph.got.rho_calculator import RhoCalculator


__all__ = ["RhoCalculator"]
