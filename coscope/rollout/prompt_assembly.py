"""
Prompt assembly: turns (GoTNode, raw_item, ancestor conclusions) into the
concrete text prompt consumed by LLMClient.

Dispatches across reasoning_path_type so that GoT / CoT / ToT share the same
rollout engine but swap templates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from coscope.core.types import GoTNode, ReasoningPathType
from coscope.graph.got.prompt_templates import (
    GOT_PLANNER_PROMPT,
    GOT_SOLVER_PROMPT,
    GOT_VERIFIER_PROMPT_TEMPLATES,
)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def build_planner_prompt(
    *,
    raw_item: Dict[str, Any],
    node: GoTNode,
    reasoning_path_type: ReasoningPathType,
) -> str:
    question = str(raw_item.get("question", "")).strip()
    hop_count = int(raw_item.get("hop_count", 2) or 2)
    # GoT is the canonical template; CoT/ToT swap happens here (v1.1).
    return GOT_PLANNER_PROMPT.format(
        question=question,
        hop_count=hop_count,
        node_id=node.node_id,
        parent_node_ids=",".join(node.parent_node_ids),
    )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _humanize_triple(raw: str, prior_conclusions: List[str]) -> str:
    """Convert MuSiQue-style semantic triples like 'A >> R' / '#1 >> R' to
    natural English questions, substituting '#N' with prior conclusion #N."""
    s = (raw or "").strip()
    if ">>" not in s:
        return s
    # Substitute #N references.
    import re
    def _sub(m):
        idx = int(m.group(1))
        if 1 <= idx <= len(prior_conclusions):
            return f"({prior_conclusions[idx - 1]})"
        return m.group(0)
    s = re.sub(r"#(\d+)", _sub, s)
    left, _, right = s.partition(">>")
    left, right = left.strip(), right.strip()
    if left and right:
        return f"What is the {right} of {left}?"
    return s


def _sub_question_for_hop(
    raw_item: Dict[str, Any],
    hop_index: int,
    prior_conclusions: Optional[List[str]] = None,
) -> str:
    prior = prior_conclusions or []
    subqs = raw_item.get("sub_questions", []) or []
    raw = None
    for sq in subqs:
        if isinstance(sq, dict) and sq.get("hop_index") == hop_index:
            raw = str(sq.get("text", "")).strip()
            break
    if raw is None and 1 <= hop_index <= len(subqs):
        v = subqs[hop_index - 1]
        if isinstance(v, str):
            raw = v.strip()
        elif isinstance(v, dict):
            raw = str(v.get("text", "")).strip()
    if not raw:
        return str(raw_item.get("question", "")).strip()
    return _humanize_triple(raw, prior)


def _evidence_for_hop(raw_item: Dict[str, Any], hop_index: int) -> str:
    """Return per-hop curated evidence text (from task_shared_items), or ''."""
    items = raw_item.get("task_shared_items", []) or []
    for it in items:
        if isinstance(it, dict) and it.get("hop_index") == hop_index:
            return str(it.get("text", "")).strip()
    return ""


def _format_prior_conclusions(prior_conclusions: List[str]) -> str:
    if not prior_conclusions:
        return "(no ancestor conclusions yet)"
    return "\n".join(f"  - {c}" for c in prior_conclusions)


def build_solver_prompt(
    *,
    raw_item: Dict[str, Any],
    node: GoTNode,
    prior_conclusions: List[str],
    reasoning_path_type: ReasoningPathType,
) -> str:
    hop = int(node.hop_index or 0)
    sub_q = _sub_question_for_hop(raw_item, hop, prior_conclusions)
    evidence = _evidence_for_hop(raw_item, hop)
    base = GOT_SOLVER_PROMPT.format(
        node_id=node.node_id,
        hop_index=hop,
        sub_question=sub_q,
        parent_node_ids=",".join(node.parent_node_ids),
        child_node_ids=",".join(node.child_node_ids),
        prior_conclusions=_format_prior_conclusions(prior_conclusions),
    )
    if evidence:
        base += f"\nRetrieved evidence:\n  {evidence}\n"
    return base


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def build_verifier_prompt(
    *,
    raw_item: Dict[str, Any],
    node: GoTNode,
    all_conclusions: List[str],
    dataset: str,
    reasoning_path_type: ReasoningPathType,
) -> str:
    template = GOT_VERIFIER_PROMPT_TEMPLATES.get(
        dataset, GOT_VERIFIER_PROMPT_TEMPLATES.get("musique", "请核查推理图中各节点结论。")
    )
    summary = "\n".join(f"  - {c}" for c in all_conclusions) or "  - (no conclusions)"
    return (
        f"{template}\n"
        f"Question: {raw_item.get('question', '')}\n"
        f"All conclusions:\n{summary}\n\n"
        "Respond in English with a concise audit note (under 80 words). "
        "Flag any credibility, temporal, or consistency issues; otherwise state 'PASS'."
    )


# ---------------------------------------------------------------------------
# QUERY_INTENT / SCRATCH prompts (role-dispatched)
# ---------------------------------------------------------------------------


_INTENT_HEADER = (
    "Step 2: Pre-retrieval reasoning. Reason silently about WHAT evidence you need\n"
    "before issuing a retrieval query. Respond in English, under 50 words, as a\n"
    "short dash-prefixed list of search targets. No preamble.\n"
)


def build_query_intent_prompt(
    *,
    role: str,
    raw_item: Dict[str, Any],
    node: GoTNode,
    prior_conclusions: List[str],
    reasoning_path_type: ReasoningPathType,
) -> str:
    base = _INTENT_HEADER
    if role == "planner":
        return base + f"Role: planner\nQuestion: {raw_item.get('question', '')}\nOutput: a short list of topical areas to retrieve."
    if role == "solver":
        hop = int(node.hop_index or 0)
        return (
            base
            + f"Role: solver\nNode: {node.node_id} (hop {hop})\n"
            + f"Sub-question: {_sub_question_for_hop(raw_item, hop, prior_conclusions)}\n"
            + f"Ancestor conclusions:\n{_format_prior_conclusions(prior_conclusions)}\n"
            + "Output: a concise retrieval intent (what evidence + which ancestors)."
        )
    if role == "verifier":
        return base + "Role: verifier\nOutput: what provenance / consistency checks to retrieve."
    return base + f"Role: {role}\nOutput: retrieval intent."


def build_scratch_prompt(
    *,
    role: str,
    raw_item: Dict[str, Any],
    node: GoTNode,
    prior_conclusions: List[str],
    reasoning_path_type: ReasoningPathType,
) -> str:
    header = (
        "Post-retrieval private reasoning. Think step-by-step before committing to a conclusion.\n"
        "Respond in English, under 150 words, as a short numbered list of reasoning steps.\n"
        "No markdown headings, no preamble.\n"
    )
    if role == "solver":
        hop = int(node.hop_index or 0)
        sub_q = _sub_question_for_hop(raw_item, hop, prior_conclusions)
        evidence = _evidence_for_hop(raw_item, hop)
        body = (
            f"Node: {node.node_id} (hop {hop})\n"
            f"Sub-question: {sub_q}\n"
            f"Ancestor conclusions:\n{_format_prior_conclusions(prior_conclusions)}\n"
        )
        if evidence:
            body += f"Retrieved evidence:\n  {evidence}\n"
        body += "Draft your reasoning; this will be kept private."
        return header + body
    return header + f"Role: {role}\nDraft private reasoning."
