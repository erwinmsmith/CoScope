"""
ToT (Tree-of-Thought) prompt templates — internal implementation.

Per `coscope/data/docs/GoT_Dataset_Requirements_v2.md` §18.5, ToT prompts are maintained
inside the project. External ToT implementations (e.g. princeton-nlp/
tree-of-thought-llm) MUST NOT be imported at runtime; their prompt strings have
been vendored and adapted here to the CoScope Planner / Solver / Verifier /
Evaluator agent role schema.

Reference: Yao et al. 2023, "Tree of Thoughts: Deliberate Problem Solving with
Large Language Models" (NeurIPS 2023). Wording adapted, not verbatim.

Unique to ToT:
    - Multiple independent branches fan out from Planner and never merge.
    - An extra Evaluator prompt (TOT_EVALUATOR_PROMPT) lets the Planner pick
      the best branch after all Solvers have produced conclusions.

Placeholder contract (see §18.6, plus ToT-specific fields):
    {question}            — raw user question (Planner)
    {branch_count}        — number of independent hypothesis branches (Planner / Evaluator)
    {depth}               — steps per branch (Planner)
    {branch_id}           — branch label, e.g. "A", "B" (Solver)
    {branch_goal}         — hypothesis goal for this branch (Solver)
    {hop_index}           — 1-based step index within a branch (Solver)
    {sub_question}        — per-step sub-question (Solver)
    {prior_conclusions}   — conclusions from earlier steps within the same branch
    {branch_summaries}    — multi-branch summaries fed to the Evaluator
"""

from coscope.prompts.graph.tot import (
    TOT_PLANNER_PROMPT,
    TOT_SOLVER_PROMPT,
    TOT_EVALUATOR_PROMPT,
    TOT_VERIFIER_PROMPT_TEMPLATES,
)


__all__ = [
    "TOT_PLANNER_PROMPT",
    "TOT_SOLVER_PROMPT",
    "TOT_EVALUATOR_PROMPT",
    "TOT_VERIFIER_PROMPT_TEMPLATES",
]
