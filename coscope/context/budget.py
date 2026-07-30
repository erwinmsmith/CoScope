"""Simple deterministic context budgeting."""

from coscope.core.memory import MemoryEntry


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class ContextBudget:
    def select(
        self,
        entries: list[MemoryEntry],
        *,
        token_budget: int,
    ) -> tuple[list[MemoryEntry], int]:
        selected: list[MemoryEntry] = []
        used = 0
        for entry in entries:
            cost = estimate_tokens(entry.content)
            if used + cost > token_budget:
                continue
            selected.append(entry)
            used += cost
        return selected, used
