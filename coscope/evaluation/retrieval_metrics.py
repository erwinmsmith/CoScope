"""Retrieval metrics for replay and live runs."""

from __future__ import annotations

from coscope.retrieval import RetrievalResult


def recall_at_k(
    results: dict[str, RetrievalResult],
    gold_by_request: dict[str, set[str]],
    *,
    k: int,
) -> float:
    scores = []
    for request_id, gold in gold_by_request.items():
        if not gold or request_id not in results:
            continue
        found = {
            candidate.memory.memory_id
            for candidate in results[request_id].candidates[:k]
        }
        scores.append(len(found & gold) / len(gold))
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(
    results: dict[str, RetrievalResult],
    gold_by_request: dict[str, set[str]],
    *,
    k: int,
) -> float:
    scores = []
    for request_id, gold in gold_by_request.items():
        if not gold or request_id not in results:
            continue
        score = 0.0
        for rank, candidate in enumerate(results[request_id].candidates[:k], start=1):
            if candidate.memory.memory_id in gold:
                score = 1.0 / rank
                break
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0
