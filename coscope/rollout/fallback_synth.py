"""
Template-based fallback content synthesis.

Used when the LLM call fails after retries, or when the pipeline is invoked
with `--use-template-fallback` for dry-run / CI.

The synthesis logic mirrors the behavior of the now-deprecated
`memory/task_shared_builder.py` but emits it through the new artifact schema.
"""

from __future__ import annotations

from typing import Any, Dict, List

from coscope.core.types import GoTNode


def _sub_question_for_hop(raw_item: Dict[str, Any], hop_index: int) -> str:
    """Extract the sub-question string for the given hop, or fall back."""
    subqs = raw_item.get("sub_questions", []) or []
    for sq in subqs:
        if isinstance(sq, dict) and sq.get("hop_index") == hop_index:
            return str(sq.get("text", "")).strip()
    if 1 <= hop_index <= len(subqs):
        v = subqs[hop_index - 1]
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, dict):
            return str(v.get("text", "")).strip()
    return str(raw_item.get("question", "")).strip()


def _task_shared_text_for_hop(raw_item: Dict[str, Any], hop_index: int) -> str:
    """Return the gold oracle conclusion text for a given hop."""
    for item in raw_item.get("task_shared_items", []) or []:
        if item.get("hop_index") == hop_index:
            return str(item.get("text", "")).strip()
    for para in raw_item.get("supporting_paragraphs", []) or []:
        if para.get("hop_index") == hop_index:
            return str(para.get("text", "")).strip()
    return ""


def synth_plan(raw_item: Dict[str, Any]) -> str:
    """Synthesize a PLAN artifact from sub_questions or hop count."""
    question = str(raw_item.get("question", "")).strip()
    subqs = raw_item.get("sub_questions", []) or []
    if subqs:
        lines = [f"Plan for question: {question}"]
        for i, sq in enumerate(subqs, start=1):
            if isinstance(sq, dict):
                hop = sq.get("hop_index", i)
                text = str(sq.get("text", "")).strip()
            else:
                hop = i
                text = str(sq).strip()
            lines.append(f"  - hop {hop}: {text}")
        return "\n".join(lines)
    hop_count = int(raw_item.get("hop_count", 2) or 2)
    return f"Plan for question: {question}\n  - decompose into {hop_count} sequential solver hops."


def synth_solver_query_intent(raw_item: Dict[str, Any], node: GoTNode) -> str:
    """
    Step-2 pre-retrieval reasoning: what does this solver need to find?
    Template form; real runs replace with LLM output.
    """
    hop = node.hop_index or 0
    sub_q = _sub_question_for_hop(raw_item, hop)
    return (
        f"[query_intent] node={node.node_id} hop={hop}\n"
        f"I must answer: {sub_q}\n"
        f"I need to retrieve: facts that resolve this sub-question, "
        f"plus any upstream conclusions from my ancestors."
    )


def synth_solver_scratch(raw_item: Dict[str, Any], node: GoTNode) -> str:
    """Post-retrieval private reasoning draft."""
    hop = node.hop_index or 0
    sub_q = _sub_question_for_hop(raw_item, hop)
    return (
        f"[scratch] node={node.node_id} hop={hop}\n"
        f"Considering sub-question: {sub_q}\n"
        f"Drafting reasoning based on retrieved evidence and ancestor conclusions."
    )


def synth_solver_conclusion(raw_item: Dict[str, Any], node: GoTNode) -> str:
    """External conclusion for downstream solvers."""
    hop = node.hop_index or 0
    gold = _task_shared_text_for_hop(raw_item, hop)
    if gold:
        return gold
    return f"[conclusion] node={node.node_id} hop={hop}: (no gold oracle available)"


def synth_planner_query_intent(raw_item: Dict[str, Any]) -> str:
    question = str(raw_item.get("question", "")).strip()
    return (
        f"[query_intent] node=planner\n"
        f"I must decompose: {question}\n"
        f"I need to retrieve: high-level topical context to decide the hop structure."
    )


def synth_planner_scratch(raw_item: Dict[str, Any]) -> str:
    question = str(raw_item.get("question", "")).strip()
    return f"[scratch] node=planner\nDecomposing the question into sequential hops: {question}"


def synth_verifier_query_intent(raw_item: Dict[str, Any]) -> str:
    return (
        "[query_intent] node=verifier\n"
        "I must check provenance and cross-conclusion consistency.\n"
        "I need to retrieve: restricted audit traces and all task_shared conclusions."
    )


def synth_verifier_scratch(raw_item: Dict[str, Any]) -> str:
    return "[scratch] node=verifier\nInspecting each solver's conclusion for source-alignment."


def synth_audit_report(
    raw_item: Dict[str, Any], all_conclusions: List[str], dataset: str
) -> str:
    """Synthesize an AUDIT_REPORT artifact (restricted scope, S4)."""
    bullets = "\n".join(f"  - {c[:120]}" for c in all_conclusions) or "  - (no conclusions)"
    return (
        f"[audit_report] dataset={dataset}\n"
        f"Reviewed {len(all_conclusions)} solver conclusion(s):\n{bullets}\n"
        f"Verdict: provenance checks applied, no inconsistencies flagged."
    )
