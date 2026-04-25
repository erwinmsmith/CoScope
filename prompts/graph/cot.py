"""
CoT (Chain-of-Thought) prompt template strings — canonical source.

Verifier templates are shared with GoT (single source of truth) to isolate
the ablation variable to planning structure only.
"""

from __future__ import annotations

from typing import Dict

from prompts.graph.got import GOT_VERIFIER_PROMPT_TEMPLATES


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


COT_VERIFIER_PROMPT_TEMPLATES: Dict[str, str] = dict(GOT_VERIFIER_PROMPT_TEMPLATES)


__all__ = [
    "COT_PLANNER_PROMPT",
    "COT_SOLVER_PROMPT",
    "COT_VERIFIER_PROMPT_TEMPLATES",
]
