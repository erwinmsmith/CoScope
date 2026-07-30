"""Final context injection decision."""

from __future__ import annotations

from coscope.context.budget import ContextBudget
from coscope.context.conflict import resolve_conflicts
from coscope.context.dedup import deduplicate
from coscope.context.pollution import PollutionGuard
from coscope.core.memory import MemoryEntry


class ContextSelector:
    def __init__(
        self,
        pollution_guard: PollutionGuard | None = None,
        budget: ContextBudget | None = None,
    ):
        self.pollution_guard = pollution_guard or PollutionGuard()
        self.budget = budget or ContextBudget()

    def select(
        self,
        entries: list[MemoryEntry],
        *,
        agent_id: str,
        token_budget: int,
    ) -> tuple[list[MemoryEntry], int]:
        allowed = [
            entry
            for entry in entries
            if self.pollution_guard.allow(entry, agent_id)
        ]
        cleaned = resolve_conflicts(deduplicate(allowed))
        return self.budget.select(cleaned, token_budget=token_budget)
