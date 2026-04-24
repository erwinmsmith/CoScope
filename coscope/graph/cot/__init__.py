"""CoT (Chain-of-Thought) module: prompt templates, chain builder, rho calculator."""

from coscope.graph.cot.chain_builder import ChainBuilder
from coscope.graph.cot.prompt_templates import (
    COT_PLANNER_PROMPT,
    COT_SOLVER_PROMPT,
    COT_VERIFIER_PROMPT_TEMPLATES,
)
from coscope.graph.cot.rho_calculator import RhoCalculator


__all__ = [
    "ChainBuilder",
    "RhoCalculator",
    "COT_PLANNER_PROMPT",
    "COT_SOLVER_PROMPT",
    "COT_VERIFIER_PROMPT_TEMPLATES",
]
