"""
ToT (Tree-of-Thought) prompt template strings — canonical source.

Verifier templates are shared with GoT (single source of truth).
"""

from __future__ import annotations

from typing import Dict

from prompts.graph.got import GOT_VERIFIER_PROMPT_TEMPLATES


TOT_PLANNER_PROMPT = """\
你是一个树状推理规划者。给定问题：{question}
请生成 {branch_count} 条独立的推理假设路径，每条路径包含 {depth} 个步骤。
各路径相互独立，代表不同的推理假设，最终选择最优路径。
输出格式：
路径A：假设目标 → 步骤A1 → 步骤A2 → ...
路径B：假设目标 → 步骤B1 → 步骤B2 → ...
"""


TOT_SOLVER_PROMPT = """\
你是推理树路径 {branch_id} 第 {hop_index} 步的执行者。
当前路径假设目标：{branch_goal}
当前子问题：{sub_question}
本路径前序结论（步骤1到步骤{hop_index_minus_one}）：
{prior_conclusions}
请基于检索到的证据，推进当前路径的推理。
注意：你的结论仅在本路径内有效，不影响其他路径。
"""


TOT_EVALUATOR_PROMPT = """\
请评估以下 {branch_count} 条推理路径，选择最可能正确的路径：
{branch_summaries}
评估标准：证据充分性（权重0.4）、逻辑一致性（权重0.4）、答案置信度（权重0.2）。
输出：最优路径ID + 选择理由
"""


TOT_VERIFIER_PROMPT_TEMPLATES: Dict[str, str] = dict(GOT_VERIFIER_PROMPT_TEMPLATES)


__all__ = [
    "TOT_PLANNER_PROMPT",
    "TOT_SOLVER_PROMPT",
    "TOT_EVALUATOR_PROMPT",
    "TOT_VERIFIER_PROMPT_TEMPLATES",
]
