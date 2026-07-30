"""Content-level context deduplication."""

from __future__ import annotations

import re

from coscope.core.memory import MemoryEntry


def deduplicate(entries: list[MemoryEntry]) -> list[MemoryEntry]:
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    output: list[MemoryEntry] = []
    for entry in entries:
        normalized = re.sub(r"\s+", " ", entry.content.casefold()).strip()
        if entry.memory_id in seen_ids or normalized in seen_content:
            continue
        seen_ids.add(entry.memory_id)
        seen_content.add(normalized)
        output.append(entry)
    return output
