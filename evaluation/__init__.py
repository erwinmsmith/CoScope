"""Evaluation utilities for CoScope retrieval experiments."""

from evaluation.metrics import (
    EvaluationReport,
    evaluate_retrieval,
    false_merge_rate,
    first_stage_savings,
    mrr_at_k,
    recall_at_k,
)
from evaluation.runner import (
    DEFAULT_VARIANTS,
    VariantRun,
    evaluate_variants,
    format_variant_table,
)
from evaluation.synthetic import (
    SyntheticCase,
    SyntheticVariantSummary,
    build_synthetic_suite,
    evaluate_synthetic_suite,
    format_synthetic_case_tables,
    format_synthetic_table,
)
from evaluation.jsonl_runner import (
    EpisodeRun,
    StratifiedCell,
    StratifiedReport,
    evaluate_jsonl,
    format_stratified_table,
    load_episodes,
)

__all__ = [
    "EvaluationReport",
    "evaluate_retrieval",
    "false_merge_rate",
    "first_stage_savings",
    "mrr_at_k",
    "recall_at_k",
    "DEFAULT_VARIANTS",
    "VariantRun",
    "evaluate_variants",
    "format_variant_table",
    "SyntheticCase",
    "SyntheticVariantSummary",
    "build_synthetic_suite",
    "evaluate_synthetic_suite",
    "format_synthetic_case_tables",
    "format_synthetic_table",
    "EpisodeRun",
    "StratifiedCell",
    "StratifiedReport",
    "evaluate_jsonl",
    "format_stratified_table",
    "load_episodes",
]
