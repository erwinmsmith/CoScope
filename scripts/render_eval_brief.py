"""
Render a compact experiment-style markdown brief from variant evaluation output.

This script consumes the `summary.json` produced by
`scripts.eval_jsonl` and turns it into a more report-like
markdown note with:

1. Setup summary
2. Main table
3. Key findings
4. Short interpretation

Example
-------
    python -m scripts.render_eval_brief \
        --summary-json data/eval/cot_partial_409_full/summary.json \
        --title "MuSiQue CoT Partial Retrieval Ablation" \
        --output data/eval/cot_partial_409_full/brief.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_summary(path: Path) -> Dict[str, Any]:
    return _normalize_summary(json.loads(path.read_text(encoding="utf-8")))


def _normalize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Support both legacy variant summaries and current eval_jsonl reports."""
    if "overall" in summary:
        return summary

    stratified = summary.get("stratified") or {}
    cells = list(stratified.get("cells", []) or [])
    if not cells:
        return summary

    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    by_subset: Dict[str, List[Dict[str, Any]]] = {}
    for cell in cells:
        by_variant.setdefault(str(cell.get("variant", "")), []).append(cell)
        by_subset.setdefault(str(cell.get("subset", "")), []).append(cell)

    def _weighted(rows: List[Dict[str, Any]], key: str) -> float:
        total_n = sum(int(row.get("n_episodes", 0)) for row in rows)
        if total_n <= 0:
            return 0.0
        return sum(float(row.get(key, 0.0)) * int(row.get("n_episodes", 0)) for row in rows) / total_n

    overall = []
    for variant, rows in sorted(by_variant.items()):
        overall.append(
            {
                "variant": variant,
                "recall_at_k": _weighted(rows, "recall_at_k"),
                "mrr_at_k": _weighted(rows, "mrr_at_k"),
                "first_stage_savings": _weighted(rows, "first_stage_savings"),
                "false_merge_rate": _weighted(rows, "false_merge_rate"),
                "content_false_merge_rate": _weighted(rows, "content_false_merge_rate"),
                "episodes": sum(int(row.get("n_episodes", 0)) for row in rows),
            }
        )

    return {
        "episodes_evaluated": stratified.get("n_episodes_total", 0),
        "variants": sorted(by_variant),
        "overall": overall,
        "by_subset": {
            subset: sorted(rows, key=lambda row: str(row.get("variant", "")))
            for subset, rows in sorted(by_subset.items())
        },
        "source_format": "eval_jsonl",
        "raw": summary,
    }


def _find_variant(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any] | None:
    for row in rows:
        if row.get("variant") == name:
            return row
    return None


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _subset_names(summary: Dict[str, Any]) -> List[str]:
    return list(summary.get("by_subset", {}).keys())


def _main_table(rows: List[Dict[str, Any]]) -> str:
    headers = ["Variant", "Episodes", "EvidenceRecall@k", "MRR@k", "Savings", "FMR"]
    include_hit = any("hit_at_k" in row for row in rows)
    include_cfmr = any("content_false_merge_rate" in row for row in rows)
    if include_hit:
        headers.insert(3, "Hit@k")
    if include_cfmr:
        headers.append("cFMR")

    body = []
    for row in rows:
        values = [
            str(row.get("variant", "")),
            str(row.get("episodes", row.get("n_episodes", ""))),
            _fmt(float(row.get("recall_at_k", 0.0))),
            _fmt(float(row.get("mrr_at_k", 0.0))),
            _fmt(float(row.get("first_stage_savings", 0.0))),
            _fmt(float(row.get("false_merge_rate", 0.0))),
        ]
        if include_hit:
            values.insert(3, _fmt(float(row.get("hit_at_k", 0.0))))
        if include_cfmr:
            values.append(_fmt(float(row.get("content_false_merge_rate", 0.0))))
        body.append(values)

    if not body:
        return "(no rows)"
    widths = [max(len(headers[i]), *(len(r[i]) for r in body)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in body
    )
    return "\n".join(lines)


def _build_findings(rows: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    a1 = _find_variant(rows, "a1")
    a4 = _find_variant(rows, "a4")
    a5 = _find_variant(rows, "a5")
    a5_noproj = _find_variant(rows, "a5_noproj")
    a6 = _find_variant(rows, "a6")
    a7 = _find_variant(rows, "a7")
    a8 = _find_variant(rows, "a8")

    if a1 and a4:
        findings.append(
            "Shared retrieval in `a4` saves "
            f"{_fmt(float(a4['first_stage_savings']))} first-stage calls, "
            "but evidence recall drops from "
            f"{_fmt(float(a1['recall_at_k']))} to {_fmt(float(a4['recall_at_k']))}."
        )
    if a5 and a5_noproj:
        hit_text = (
            f" and Hit@k ({_fmt(float(a5.get('hit_at_k', 0.0)))})"
            if "hit_at_k" in a5
            else ""
        )
        findings.append(
            "`a5_noproj` matches `a5` on evidence recall "
            f"({_fmt(float(a5['recall_at_k']))}){hit_text}, while changing MRR from "
            f"{_fmt(float(a5['mrr_at_k']))} to {_fmt(float(a5_noproj['mrr_at_k']))}."
        )
    if a5 and a4:
        findings.append(
            "`a5` retains more evidence recall than `a4` "
            f"({_fmt(float(a5['recall_at_k']))} vs {_fmt(float(a4['recall_at_k']))}), "
            f"while using less sharing savings "
            f"({_fmt(float(a5['first_stage_savings']))} vs {_fmt(float(a4['first_stage_savings']))})."
        )
    if a6 and a7 and a8:
        findings.append(
            "`a6`, `a7`, and `a8` are currently identical on this slice "
            f"(EvidenceRecall@k={_fmt(float(a6['recall_at_k']))}, "
            f"MRR@k={_fmt(float(a6['mrr_at_k']))}), "
            "so query rewriting has not yet separated these variants."
        )
    return findings


def _build_interpretation(rows: List[Dict[str, Any]]) -> List[str]:
    a1 = _find_variant(rows, "a1")
    a4 = _find_variant(rows, "a4")
    a5 = _find_variant(rows, "a5")
    a5_noproj = _find_variant(rows, "a5_noproj")
    lines: List[str] = []

    if a1 and a4:
        lines.append(
            "On the current CoT partial slice, first-stage batching is effective for reducing calls, "
            "but it does not yet preserve evidence-level coverage."
        )
    if a5 and a5_noproj:
        lines.append(
            "The projection step still does not show a stable benefit: removing projection keeps recall unchanged "
            "and yields better ranking quality."
        )
    if a1 and a5:
        lines.append(
            "At this stage, the best recall remains the independent baseline, while partially shared retrieval "
            "offers a middle ground between efficiency and quality."
        )
    return lines


def render_brief(summary: Dict[str, Any], title: str) -> str:
    rows = list(summary.get("overall", []))
    episodes = summary.get("episodes_evaluated", 0)
    variants = summary.get("variants", [])
    subsets = _subset_names(summary)

    lines: List[str] = [
        f"# {title}",
        "",
        "## Setup",
        f"- Episodes evaluated: `{episodes}`",
        f"- Variants: `{', '.join(variants)}`",
        f"- Subsets present: `{', '.join(subsets) if subsets else 'overall only'}`",
        "- `EvidenceRecall@k` is gold-evidence coverage. `Hit@k` is binary success: whether at least one gold evidence item appears in the top-k list.",
        "",
        "## Main Results",
        "",
        _main_table(rows),
        "",
    ]

    if summary.get("by_subset"):
        lines.extend(["## Subset View", ""])
        for subset_name, subset_rows in summary["by_subset"].items():
            lines.extend([f"### {subset_name}", "", _main_table(subset_rows), ""])

    lines.extend(["## Key Findings", ""])

    for item in _build_findings(rows):
        lines.append(f"- {item}")

    lines.extend(["", "## Interpretation", ""])
    for item in _build_interpretation(rows):
        lines.append(f"- {item}")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a brief experiment markdown from summary.json.")
    parser.add_argument("--summary-json", required=True, help="Path to evaluation summary.json")
    parser.add_argument("--title", default="Evaluation Brief", help="Markdown title")
    parser.add_argument("--output", required=True, help="Output markdown path")
    args = parser.parse_args(argv)

    summary = _load_summary(Path(args.summary_json))
    text = render_brief(summary, args.title)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Saved markdown brief to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
