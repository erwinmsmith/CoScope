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

from __future__ import annotations

from typing import Dict


# ------------------------------------------------------------------
# Planner / Solver prompts
# ------------------------------------------------------------------


GOT_PLANNER_PROMPT = """\
You are a graph-of-thought planner. Given the question: {question}
Decompose it into a reasoning graph with exactly {hop_count} nodes.
Each node has a sub-question; edges encode dependency ordering.

OUTPUT FORMAT (one line per node, English only, no extra commentary):
Node {node_id} (parents: {parent_node_ids}): <sub-question>

Constraints:
- Respond in English.
- One line per node. No bullet lists, no markdown, no preamble.
- Keep each sub-question under 20 words.
"""


GOT_SOLVER_PROMPT = """\
You are the solver at graph node {node_id} (hop {hop_index}).

Sub-question: {sub_question}

Ancestor conclusions (from parents {parent_node_ids}):
{prior_conclusions}

Produce a CONCLUSION for this node that directly answers the sub-question.
Your conclusion will be inherited by child nodes {child_node_ids}.

Constraints:
- Respond in English.
- Output one or two sentences, under 60 words.
- Directly answer the sub-question; do not restate the task.
- No markdown, no bullet lists, no preamble like "The answer is:".
"""


# ------------------------------------------------------------------
# Verifier prompts (dataset-typed)
# ------------------------------------------------------------------


GOT_VERIFIER_PROMPT_TEMPLATES: Dict[str, str] = {
    "musique":                       "Audit each node's conclusion for source credibility and temporal consistency.",
    "hotpotqa":                      "Audit each node's conclusion for source credibility and temporal consistency.",
    "2wikimhqa_bridge":              "Audit bridge-hop conclusions for source credibility and temporal consistency.",
    "2wikimhqa_comparison":          "Audit whether the two compared entities rely on authoritative, non-conflicting sources.",
    "2wikimhqa_bridge_comparison":   "Audit the logical consistency of reasoning steps and the completeness of evidence.",
    "gsm8k":                         "Verify that each step's numeric result matches the gold intermediate value.",
    "math_algebra":                  "Verify algebraic transformations and that numeric results are self-consistent.",
    "math_geometry":                 "Verify geometric relations and that cited properties are applied correctly.",
    "math_number_theory":            "Verify applications of divisibility, congruence, and related properties.",
    "math_counting_and_probability": "Verify completeness of the sample space and accuracy of probability calculations.",
    "math_intermediate_algebra":     "Verify algebraic transformations and equation-solving steps.",
    "math_prealgebra":               "Verify basic arithmetic and quantitative relationships.",
    "math_precalculus":              "Verify applications of functions, limits, and trigonometric identities.",
}


__all__ = [
    "GOT_PLANNER_PROMPT",
    "GOT_SOLVER_PROMPT",
    "GOT_VERIFIER_PROMPT_TEMPLATES",
]
