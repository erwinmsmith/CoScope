"""
GoT (Graph-of-Thought) prompt templates — internal implementation.

Per `data/docs/GoT_Dataset_Requirements_v2.md` §18.3, prompts for the three
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

from __future__ import annotations

from typing import Dict


# ------------------------------------------------------------------
# Planner / Solver prompts
# ------------------------------------------------------------------


GOT_PLANNER_PROMPT = """\
你是一个图状推理规划者。给定问题：{question}
请将问题分解为一个推理图，包含 {hop_count} 个推理节点。
每个节点对应一个子问题，节点间的依赖关系决定推理顺序。
输出格式（每行一个节点）：
节点{node_id}（依赖：{parent_node_ids}）：<子问题>
"""


GOT_SOLVER_PROMPT = """\
你是推理图节点 {node_id}（第 {hop_index} 跳）的执行者。
当前子问题：{sub_question}
可用的祖先节点结论（来自节点 {parent_node_ids}）：
{prior_conclusions}
请基于检索到的证据和祖先结论，给出当前节点的推理结论。
结论将被后继节点（{child_node_ids}）继承。
"""


# ------------------------------------------------------------------
# Verifier prompts (dataset-typed)
# ------------------------------------------------------------------


GOT_VERIFIER_PROMPT_TEMPLATES: Dict[str, str] = {
    "musique":                       "请核查推理图中各节点结论的来源可信度和时间一致性。",
    "hotpotqa":                      "请核查推理图中各节点结论的来源可信度和时间一致性。",
    "2wikimhqa_bridge":              "请核查推理链中各跳结论的来源可信度和时间一致性。",
    "2wikimhqa_comparison":          "请核查两个实体的对比证据来源是否权威且无冲突。",
    "2wikimhqa_bridge_comparison":   "请核查推理步骤的逻辑一致性和证据完整性。",
    "gsm8k":                         "请核查各步骤的数值计算结果是否与 gold 中间值一致。",
    "math_algebra":                  "请核查各步骤的等式变换是否一致，数值结果是否自洽。",
    "math_geometry":                 "请核查图形关系和几何性质的引用是否正确。",
    "math_number_theory":            "请核查整除性、同余等性质的应用是否正确。",
    "math_counting_and_probability": "请核查样本空间的完整性和概率计算的准确性。",
    "math_intermediate_algebra":     "请核查代数变换步骤和方程求解过程的一致性。",
    "math_prealgebra":               "请核查基础运算和数量关系的正确性。",
    "math_precalculus":              "请核查函数、极限、三角恒等式的应用是否正确。",
}


__all__ = [
    "GOT_PLANNER_PROMPT",
    "GOT_SOLVER_PROMPT",
    "GOT_VERIFIER_PROMPT_TEMPLATES",
]
