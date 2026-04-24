"""
Variant evaluation runner.

Runs the no-training ablation variants (A1/A3/A4/A5) on the same request batch
and returns comparable metric reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from coscope.core.types import RetrievalRequest, RetrievalResult
from coscope.engine import CoScope
from coscope.evaluation.metrics import EvaluationReport, GoldMap, ResultPredicate
from coscope.evaluation.metrics import evaluate_retrieval


DEFAULT_VARIANTS = ("a1", "a3", "a4", "a5")


@dataclass
class VariantRun:
    """Evaluation output for one retrieval variant."""

    variant: str
    results: List[RetrievalResult]
    report: EvaluationReport
    pipeline_stats: Mapping[str, Any]

    def to_row(self) -> dict[str, Any]:
        """Return the compact row used by comparison tables."""
        return {
            "variant": self.variant,
            "recall_at_k": self.report.recall_at_k,
            "mrr_at_k": self.report.mrr_at_k,
            "first_stage_savings": self.report.first_stage_savings,
            "false_merge_rate": self.report.false_merge_rate,
            "first_stage_actual": self.report.first_stage_actual,
            "first_stage_independent": self.report.first_stage_independent,
            "shareable_buckets": self.pipeline_stats.get("shareable_buckets", 0),
            "independent_requests": self.pipeline_stats.get("independent_requests", 0),
            "fallback_triggers": self.pipeline_stats.get("fallback_triggers", 0),
        }


def evaluate_variants(
    coscope: CoScope,
    requests: Sequence[RetrievalRequest],
    gold_by_request: GoldMap,
    *,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    k: int = 10,
    conflict_request_ids: Optional[Iterable[str]] = None,
    conflict_predicate: Optional[ResultPredicate] = None,
    independent_first_stage: Optional[int] = None,
) -> List[VariantRun]:
    """
    Run several retrieval variants on the same requests and evaluate them.

    Args:
        coscope: Initialized CoScope engine.
        requests: Retrieval requests to evaluate.
        gold_by_request: Mapping request_id -> gold memory ids.
        variants: Experiment variants to run.
        k: Cutoff for Recall@k and MRR@k.
        conflict_request_ids: Request ids that should not be merged into shared
            buckets, typically S4/verifier requests.
        conflict_predicate: Optional predicate for selecting conflict requests.
        independent_first_stage: Optional denominator for first-stage savings.

    Returns:
        A list of VariantRun objects, one per variant.
    """
    runs: List[VariantRun] = []
    denominator = independent_first_stage if independent_first_stage is not None else len(requests)

    for variant in variants:
        coscope.data.pipeline.reset_stats()
        results = coscope.retrieve(list(requests), variant=variant)
        stats = dict(coscope.get_stats()["pipeline_stats"])
        report = evaluate_retrieval(
            results,
            gold_by_request,
            k=k,
            pipeline_stats=stats,
            independent_first_stage=denominator,
            conflict_request_ids=conflict_request_ids,
            conflict_predicate=conflict_predicate,
        )
        runs.append(
            VariantRun(
                variant=variant,
                results=list(results),
                report=report,
                pipeline_stats=stats,
            )
        )

    return runs


def format_variant_table(
    runs: Sequence[VariantRun],
    *,
    digits: int = 4,
) -> str:
    """Format variant reports as a Markdown comparison table."""
    headers = [
        "Variant",
        "Recall@k",
        "MRR@k",
        "Savings",
        "FMR",
        "First-stage",
        "Buckets",
        "Independent",
        "Fallbacks",
    ]
    rows = []
    for run in runs:
        row = run.to_row()
        rows.append(
            [
                str(row["variant"]).upper(),
                _fmt(row["recall_at_k"], digits),
                _fmt(row["mrr_at_k"], digits),
                _fmt(row["first_stage_savings"], digits),
                _fmt(row["false_merge_rate"], digits),
                f"{row['first_stage_actual']}/{row['first_stage_independent']}",
                str(row["shareable_buckets"]),
                str(row["independent_requests"]),
                str(row["fallback_triggers"]),
            ]
        )

    return _markdown_table(headers, rows)


def _fmt(value: Any, digits: int) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(
        str(value).ljust(widths[i]) for i, value in enumerate(headers)
    ) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])
