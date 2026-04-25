"""
CoT (Chain-of-Thought) prompt templates — internal implementation.

Per `docs/GoT_Dataset_Requirements_v2.md` §18.4, CoT prompts are maintained
inside the project. External implementations (e.g. langchain.prompts chain
templates, Wei et al. 2022 released prompt collections) MUST NOT be imported
at runtime. The Verifier template dictionary is kept byte-identical to the GoT
version (single source of truth) to isolate the ablation variable to planning
structure.

Reference: Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in
Large Language Models" (NeurIPS 2022). The wording here is adapted to the
CoScope Planner / Solver / Verifier agent role schema and is not a verbatim
copy of any external codebase.

Placeholder contract (see §18.6):
    {question}            — raw user question (Planner)
    {hop_count}           — number of sequential steps (Planner)
    {hop_index}           — 1-based step index (Solver)
    {sub_question}        — per-step sub-question (Solver)
    {prior_conclusions}   — formatted list of conclusions from steps 1..hop_index-1
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
