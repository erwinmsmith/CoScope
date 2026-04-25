"""
CoT rho calculator.

Per `coscope/data/docs/GoT_Dataset_Requirements_v2.md` §18.4: "ρ 计算逻辑与 GoT LINEAR
完全相同，直接调用 `got/rho_calculator.py` 的 `compute_rho`". This module
re-exports the GoT implementation so downstream code can depend on a stable
`coscope.data.cot.rho_calculator` namespace without duplicating logic.
"""

from __future__ import annotations

from graph.got.rho_calculator import RhoCalculator


__all__ = ["RhoCalculator"]
