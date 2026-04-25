"""
GoT (Graph-of-Thought) prompt templates — internal implementation.

Per `coscope/data/docs/GoT_Dataset_Requirements_v2.md` §18.3, prompts for the three
reasoning-path structures (GoT / CoT / ToT) must be fully internalized.
Importing from external prompt libraries (langchain.prompts, llama_index.prompts,
haystack.nodes.prompt, guidance, promptflow, ...) is forbidden here and enforced
by `coscope/data/scripts/check_prompt_isolation.py`.

GoT is this project's original design; no external source exists to vendor.
These strings are the canonical reference used by GoT Planner / Solver / Verifier
agents and by the ablation experiments that compare GoT against CoT / ToT.

Placeholder contract (see §18.6):
    {question}            — raw user question (all roles)
    {hop_count}           — total number of solver nodes (Planner)
    {node_id}             — solver node id (Solver)
    {hop_index}           — 1-based hop index (Solver)
    {sub_question}        — per-node sub-question (Solver)
    {parent_node_ids}     — comma-separated ancestor node ids (Solver)
    {child_node_ids}      — comma-separated successor node ids (Solver)
    {prior_conclusions}   — formatted list of ancestor conclusions (Solver)
"""

from prompts.graph.got import (
    GOT_PLANNER_PROMPT,
    GOT_SOLVER_PROMPT,
    GOT_VERIFIER_PROMPT_TEMPLATES,
)


__all__ = [
    "GOT_PLANNER_PROMPT",
    "GOT_SOLVER_PROMPT",
    "GOT_VERIFIER_PROMPT_TEMPLATES",
]
