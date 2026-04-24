"""
CoT (Chain-of-Thought) prompt templates — internal implementation.

Per `coscope/data/docs/GoT_Dataset_Requirements_v2.md` §18.4, CoT prompts are maintained
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

from __future__ import annotations

from typing import Dict

from coscope.graph.got.prompt_templates import GOT_VERIFIER_PROMPT_TEMPLATES


# ------------------------------------------------------------------
# Planner / Solver prompts
# ------------------------------------------------------------------


COT_PLANNER_PROMPT = """\
你是一个多步骤推理规划者。给定问题：{question}
请将问题分解为 {hop_count} 个顺序推理步骤，每步输出一个子问题。
步骤必须严格按顺序执行，后步依赖前步结论。
输出格式：
步骤1：<子问题>
步骤2：<子问题>
...
"""


COT_SOLVER_PROMPT = """\
你是推理链第 {hop_index} 步的执行者。
当前子问题：{sub_question}
前序步骤结论（步骤1到步骤{hop_index_minus_one}）：
{prior_conclusions}
请基于检索到的证据和前序结论，给出当前步骤的结论。
结论将传递给下一步骤。
"""


# ------------------------------------------------------------------
# Verifier prompts (dataset-typed)
# ------------------------------------------------------------------
# §18.4: CoT Verifier templates share content with GoT Verifier templates
# (same dataset keys, same wording). Re-exported here as a single source of
# truth so downstream code can import from either module.


COT_VERIFIER_PROMPT_TEMPLATES: Dict[str, str] = dict(GOT_VERIFIER_PROMPT_TEMPLATES)


__all__ = [
    "COT_PLANNER_PROMPT",
    "COT_SOLVER_PROMPT",
    "COT_VERIFIER_PROMPT_TEMPLATES",
]
