"""
Canonical prompt string source for graph-of-thought reasoning structures.

This sub-package is the single source of truth for all GoT / CoT / ToT
prompt template strings. :mod:`coscope.graph` modules import from here;
the central :mod:`coscope.prompts.registry` also bootstraps from here.

Adding a new graph type: create ``<type>.py``, expose its constants via
this ``__init__.py``, and register them in ``registry._bootstrap_builtin_prompts``.
"""

from prompts.graph.got import (
    GOT_PLANNER_PROMPT,
    GOT_SOLVER_PROMPT,
    GOT_VERIFIER_PROMPT_TEMPLATES,
)
from prompts.graph.cot import (
    COT_PLANNER_PROMPT,
    COT_SOLVER_PROMPT,
    COT_VERIFIER_PROMPT_TEMPLATES,
)
from prompts.graph.tot import (
    TOT_PLANNER_PROMPT,
    TOT_SOLVER_PROMPT,
    TOT_EVALUATOR_PROMPT,
    TOT_VERIFIER_PROMPT_TEMPLATES,
)

__all__ = [
    "GOT_PLANNER_PROMPT",
    "GOT_SOLVER_PROMPT",
    "GOT_VERIFIER_PROMPT_TEMPLATES",
    "COT_PLANNER_PROMPT",
    "COT_SOLVER_PROMPT",
    "COT_VERIFIER_PROMPT_TEMPLATES",
    "TOT_PLANNER_PROMPT",
    "TOT_SOLVER_PROMPT",
    "TOT_EVALUATOR_PROMPT",
    "TOT_VERIFIER_PROMPT_TEMPLATES",
]
