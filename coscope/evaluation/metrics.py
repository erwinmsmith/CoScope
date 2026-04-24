"""
Minimal retrieval evaluation metrics.

The functions here intentionally operate on the runtime `RetrievalResult`
objects instead of dataset-specific files. This makes them usable for toy
checks, synthetic episodes, and later full dataset evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from coscope.core.types import RetrievalResult


GoldMap = Mapping[str, Sequence[str]]
ResultPredicate = Callable[[RetrievalResult], bool]


@dataclass
class EvaluationReport:
    """Aggregated retrieval metrics for one run."""

    recall_at_k: float
    mrr_at_k: float
    first_stage_savings: float
    false_merge_rate: float
    k: int
    evaluated_requests: int
    gold_requests: int
    first_stage_actual: int
    first_stage_independent: int
    false_merge_count: int
    conflict_request_count: int
    # Content-level false merge: whether a non-verifier request's top-k
    # actually contains any restricted (verifier-only) memory. Complements
    # the routing-level ``false_merge_rate`` above, which only checks whether
    # a conflict request was placed in a shared bucket.
    content_false_merge_rate: float = 0.0
    content_false_merge_count: int = 0
    content_non_verifier_request_count: int = 0
    per_request: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "recall_at_k": self.recall_at_k,
            "mrr_at_k": self.mrr_at_k,
            "first_stage_savings": self.first_stage_savings,
            "false_merge_rate": self.false_merge_rate,
            "content_false_merge_rate": self.content_false_merge_rate,
            "k": self.k,
            "evaluated_requests": self.evaluated_requests,
            "gold_requests": self.gold_requests,
            "first_stage_actual": self.first_stage_actual,
            "first_stage_independent": self.first_stage_independent,
            "false_merge_count": self.false_merge_count,
            "conflict_request_count": self.conflict_request_count,
            "content_false_merge_count": self.content_false_merge_count,
            "content_non_verifier_request_count": self.content_non_verifier_request_count,
            "per_request": self.per_request,
        }


def recall_at_k(
    results: Sequence[RetrievalResult],
    gold_by_request: GoldMap,
    k: int = 10,
) -> float:
    """
    Compute mean evidence Recall@k.

    For each request with gold evidence, recall is:
    `|top_k_memory_ids intersection gold_ids| / |gold_ids|`.
    Requests without gold evidence are skipped.
    """
    values = _per_request_recall(results, gold_by_request, k)
    if not values:
        return 0.0
    return sum(values.values()) / len(values)


def mrr_at_k(
    results: Sequence[RetrievalResult],
    gold_by_request: GoldMap,
    k: int = 10,
) -> float:
    """
    Compute mean reciprocal rank at k.

    For each request, the score is `1 / rank` for the first retrieved gold
    evidence in the top-k list, or 0 if none is found. Requests without gold
    evidence are skipped.
    """
    values = _per_request_mrr(results, gold_by_request, k)
    if not values:
        return 0.0
    return sum(values.values()) / len(values)


def first_stage_savings(
    *,
    actual_first_stage: int,
    independent_first_stage: int,
) -> float:
    """
    Compute first-stage retrieval savings relative to independent retrieval.

    `savings = (independent - actual) / independent`.
    """
    if independent_first_stage <= 0:
        return 0.0
    saved = independent_first_stage - actual_first_stage
    return saved / independent_first_stage


def false_merge_rate(
    results: Sequence[RetrievalResult],
    conflict_request_ids: Optional[Iterable[str]] = None,
    conflict_predicate: Optional[ResultPredicate] = None,
) -> float:
    """
    Compute False Merge Rate (FMR) for policy-conflict/S4 requests.

    A conflict request is counted as false-merged when its result metadata
    indicates it participated in a shared bucket, i.e. has a `bucket_id` or a
    mode that is not independent. By default, callers pass `conflict_request_ids`
    such as verifier/S4 request ids. A predicate can be supplied instead for
    dataset-specific selection.
    """
    count, total = false_merge_counts(
        results,
        conflict_request_ids=conflict_request_ids,
        conflict_predicate=conflict_predicate,
    )
    return count / total if total else 0.0


def false_merge_counts(
    results: Sequence[RetrievalResult],
    conflict_request_ids: Optional[Iterable[str]] = None,
    conflict_predicate: Optional[ResultPredicate] = None,
) -> tuple[int, int]:
    """Return `(false_merge_count, conflict_request_count)`."""
    conflict_ids = set(conflict_request_ids or [])
    false_merges = 0
    total = 0

    for result in results:
        is_conflict = (
            conflict_predicate(result)
            if conflict_predicate is not None
            else result.request_id in conflict_ids
        )
        if not is_conflict:
            continue

        total += 1
        if _is_shared_result(result):
            false_merges += 1

    return false_merges, total


def content_false_merge_counts(
    results: Sequence[RetrievalResult],
    restricted_memory_ids: Iterable[str],
    conflict_request_ids: Optional[Iterable[str]] = None,
    conflict_predicate: Optional[ResultPredicate] = None,
    k: int = 10,
) -> tuple[int, int]:
    """
    Count content-level false merges.

    A request is counted as a content-level false merge when **it is not the
    conflict (verifier) request**, yet its top-k retrieved candidates contain
    at least one memory id in ``restricted_memory_ids`` (the verifier-only
    content, e.g. POLICY_ISOLATED audit reports).

    Returns ``(leak_count, non_conflict_request_count)``. The denominator is
    the number of **non-verifier** requests in conflict episodes only; for
    episodes without conflict (no restricted ids) it is 0 and the rate is 0.
    """
    restricted = set(restricted_memory_ids or [])
    if not restricted:
        return 0, 0

    conflict_ids = set(conflict_request_ids or [])
    leaks = 0
    non_conflict = 0

    for result in results:
        is_conflict = (
            conflict_predicate(result)
            if conflict_predicate is not None
            else result.request_id in conflict_ids
        )
        # Content-level FMR is about *non-verifier* agents seeing
        # verifier-only content. Skip the verifier's own result.
        if is_conflict:
            continue
        non_conflict += 1
        top_ids = set(_top_memory_ids(result, k))
        if top_ids & restricted:
            leaks += 1

    return leaks, non_conflict


def content_false_merge_rate(
    results: Sequence[RetrievalResult],
    restricted_memory_ids: Iterable[str],
    conflict_request_ids: Optional[Iterable[str]] = None,
    conflict_predicate: Optional[ResultPredicate] = None,
    k: int = 10,
) -> float:
    leaks, total = content_false_merge_counts(
        results,
        restricted_memory_ids,
        conflict_request_ids=conflict_request_ids,
        conflict_predicate=conflict_predicate,
        k=k,
    )
    return leaks / total if total else 0.0


def evaluate_retrieval(
    results: Sequence[RetrievalResult],
    gold_by_request: GoldMap,
    *,
    k: int = 10,
    pipeline_stats: Optional[Mapping[str, Any]] = None,
    independent_first_stage: Optional[int] = None,
    conflict_request_ids: Optional[Iterable[str]] = None,
    conflict_predicate: Optional[ResultPredicate] = None,
    restricted_memory_ids: Optional[Iterable[str]] = None,
) -> EvaluationReport:
    """Compute the minimal metric bundle for one retrieval run."""
    stats = pipeline_stats or {}
    actual_first_stage = int(stats.get("first_stage_retrievals", len(results)))
    independent_count = (
        int(independent_first_stage)
        if independent_first_stage is not None
        else len(results)
    )

    recall_values = _per_request_recall(results, gold_by_request, k)
    mrr_values = _per_request_mrr(results, gold_by_request, k)
    false_merges, conflict_count = false_merge_counts(
        results,
        conflict_request_ids=conflict_request_ids,
        conflict_predicate=conflict_predicate,
    )
    content_leaks, content_total = content_false_merge_counts(
        results,
        restricted_memory_ids or [],
        conflict_request_ids=conflict_request_ids,
        conflict_predicate=conflict_predicate,
        k=k,
    )
    restricted_set = set(restricted_memory_ids or [])

    per_request: Dict[str, Dict[str, Any]] = {}
    for result in results:
        top_ids = _top_memory_ids(result, k)
        gold = set(gold_by_request.get(result.request_id, []))
        is_conflict = _request_is_conflict(
            result, conflict_request_ids, conflict_predicate
        )
        restricted_hits = (
            sorted(set(top_ids) & restricted_set)
            if restricted_set and not is_conflict
            else []
        )
        per_request[result.request_id] = {
            "agent_id": result.agent_id,
            "mode": result.metadata.get("mode"),
            "top_k_memory_ids": top_ids,
            "gold_ids": sorted(gold),
            "hits": sorted(set(top_ids) & gold),
            "recall_at_k": recall_values.get(result.request_id),
            "mrr_at_k": mrr_values.get(result.request_id),
            "false_merged": _is_shared_result(result) if is_conflict else False,
            "content_leaked": bool(restricted_hits),
            "restricted_hits": restricted_hits,
        }

    return EvaluationReport(
        recall_at_k=(
            sum(recall_values.values()) / len(recall_values) if recall_values else 0.0
        ),
        mrr_at_k=(sum(mrr_values.values()) / len(mrr_values) if mrr_values else 0.0),
        first_stage_savings=first_stage_savings(
            actual_first_stage=actual_first_stage,
            independent_first_stage=independent_count,
        ),
        false_merge_rate=false_merges / conflict_count if conflict_count else 0.0,
        content_false_merge_rate=(
            content_leaks / content_total if content_total else 0.0
        ),
        k=k,
        evaluated_requests=len(results),
        gold_requests=len(recall_values),
        first_stage_actual=actual_first_stage,
        first_stage_independent=independent_count,
        false_merge_count=false_merges,
        conflict_request_count=conflict_count,
        content_false_merge_count=content_leaks,
        content_non_verifier_request_count=content_total,
        per_request=per_request,
    )


def _per_request_recall(
    results: Sequence[RetrievalResult],
    gold_by_request: GoldMap,
    k: int,
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for result in results:
        gold = set(gold_by_request.get(result.request_id, []))
        if not gold:
            continue
        top_ids = set(_top_memory_ids(result, k))
        values[result.request_id] = len(top_ids & gold) / len(gold)
    return values


def _per_request_mrr(
    results: Sequence[RetrievalResult],
    gold_by_request: GoldMap,
    k: int,
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for result in results:
        gold = set(gold_by_request.get(result.request_id, []))
        if not gold:
            continue
        score = 0.0
        for idx, memory_id in enumerate(_top_memory_ids(result, k), 1):
            if memory_id in gold:
                score = 1.0 / idx
                break
        values[result.request_id] = score
    return values


def _top_memory_ids(result: RetrievalResult, k: int) -> List[str]:
    return [candidate.memory.memory_id for candidate in result.candidates[:k]]


def _is_shared_result(result: RetrievalResult) -> bool:
    mode = str(result.metadata.get("mode", ""))
    if result.metadata.get("bucket_id"):
        return True
    return "independent" not in mode


def _request_is_conflict(
    result: RetrievalResult,
    conflict_request_ids: Optional[Iterable[str]],
    conflict_predicate: Optional[ResultPredicate],
) -> bool:
    if conflict_predicate is not None:
        return bool(conflict_predicate(result))
    return result.request_id in set(conflict_request_ids or [])
