"""Stage-specific unauthorized exposure counters."""

from dataclasses import dataclass


@dataclass
class SafetyReport:
    unauthorized_corpus_access: int = 0
    unauthorized_candidate_exposure: int = 0
    unauthorized_context_exposure: int = 0
    cross_branch_leakage: int = 0
    cache_leakage: int = 0
    log_leakage: int = 0

    @property
    def unauthorized_exposure(self) -> int:
        return sum(vars(self).values())
