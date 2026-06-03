"""Compatibility re-export for legacy CoT prompt imports.

Canonical CoT prompts live in :mod:`prompts.graph.cot`. This module preserves
the old ``graph.cot.prompt_templates`` import path used by earlier CoT scripts.
"""

from prompts.graph.cot import (
    COT_PLANNER_PROMPT,
    COT_SOLVER_PROMPT,
    COT_VERIFIER_PROMPT_TEMPLATES,
)

__all__ = [
    "COT_PLANNER_PROMPT",
    "COT_SOLVER_PROMPT",
    "COT_VERIFIER_PROMPT_TEMPLATES",
]
