"""Runtime efficiency metrics."""

from dataclasses import asdict

from coscope.retrieval import RetrievalStats


def retrieval_savings(stats: RetrievalStats) -> float:
    if not stats.requests:
        return 0.0
    return 1.0 - stats.shared_store_queries / stats.requests


def runtime_report(stats: RetrievalStats) -> dict[str, int | float]:
    return {**asdict(stats), "retrieval_savings": retrieval_savings(stats)}
