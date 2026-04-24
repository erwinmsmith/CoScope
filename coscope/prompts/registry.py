"""
Central prompt registry.

All prompt strings used by the rollout engine, the graph builders (GoT/CoT/ToT)
and any other subsystem should be registered here under a stable key. The
registry exposes a pluggable backend so that the default in-memory dict
backend can later be swapped for a database-backed backend without touching
call sites.

Keys follow a namespaced convention:

    <subsystem>/<graph_type>/<role>[:<dataset>]

Examples:

    rollout/step2/header                -- Step-2 pre-retrieval header
    rollout/scratch/header              -- post-retrieval scratch header
    graph/got/planner                   -- GoT planner prompt
    graph/got/solver                    -- GoT solver prompt
    graph/got/verifier:musique          -- GoT verifier template (MuSiQue)
    graph/cot/solver                    -- CoT solver prompt
    graph/tot/solver                    -- ToT solver prompt

Call sites should read through ``get_prompt(key)`` rather than hard-coding
strings, so the underlying storage can be swapped later.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Protocol


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class PromptBackend(Protocol):
    """Protocol a prompt backend must satisfy. Swap for a DB-backed
    implementation to persist prompts outside the codebase."""

    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str) -> None: ...
    def keys(self) -> Iterable[str]: ...
    def contains(self, key: str) -> bool: ...


class InMemoryBackend:
    """Default backend: a plain Python dict, populated at module import."""

    def __init__(self) -> None:
        self._data: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def keys(self) -> Iterable[str]:
        return list(self._data.keys())

    def contains(self, key: str) -> bool:
        return key in self._data


# ---------------------------------------------------------------------------
# Registry singleton
# ---------------------------------------------------------------------------


class PromptRegistry:
    """Facade over a PromptBackend; exposes a small stable surface."""

    def __init__(self, backend: Optional[PromptBackend] = None) -> None:
        self._backend: PromptBackend = backend or InMemoryBackend()

    def set_backend(self, backend: PromptBackend) -> None:
        """Replace the backend. Intended for the DB-integration path."""
        self._backend = backend

    def register(self, key: str, value: str, *, overwrite: bool = False) -> None:
        if not overwrite and self._backend.contains(key):
            return
        self._backend.set(key, value)

    def get(self, key: str, default: Optional[str] = None) -> str:
        value = self._backend.get(key)
        if value is None:
            if default is not None:
                return default
            raise KeyError(f"Prompt key not registered: {key!r}")
        return value

    def has(self, key: str) -> bool:
        return self._backend.contains(key)

    def keys(self) -> Iterable[str]:
        return self._backend.keys()


# Module-level singleton. Import and use ``get_prompt`` / ``register_prompt``.
_registry = PromptRegistry()


def register_prompt(key: str, value: str, *, overwrite: bool = False) -> None:
    """Register a prompt string under ``key``."""
    _registry.register(key, value, overwrite=overwrite)


def get_prompt(key: str, default: Optional[str] = None) -> str:
    """Fetch a prompt string by key; raises ``KeyError`` if missing and no
    default is provided."""
    return _registry.get(key, default=default)


def has_prompt(key: str) -> bool:
    return _registry.has(key)


def set_backend(backend: PromptBackend) -> None:
    """Install a custom backend (e.g. a DB-backed one)."""
    _registry.set_backend(backend)


def list_keys() -> Iterable[str]:
    return _registry.keys()


# ---------------------------------------------------------------------------
# Eager registration of built-in prompts
# ---------------------------------------------------------------------------
#
# We register the built-in prompts at import time from the existing source
# modules so that legacy call sites continue to work unchanged. External
# integrations can override any key via ``register_prompt(..., overwrite=True)``
# or by installing a custom backend before these registrations run.


def _bootstrap_builtin_prompts() -> None:
    """Pull prompts from the legacy source-of-truth modules into the registry.

    This function is idempotent and defensive: any import failure is logged
    and skipped so the registry never blocks module import.
    """
    # Graph prompts (GoT, CoT, ToT). Each graph module is the source of truth
    # for its own template strings; we mirror them into the registry.
    try:
        from coscope.graph.got import prompt_templates as got_pt

        register_prompt("graph/got/planner", got_pt.GOT_PLANNER_PROMPT)
        register_prompt("graph/got/solver", got_pt.GOT_SOLVER_PROMPT)
        for dataset, text in got_pt.GOT_VERIFIER_PROMPT_TEMPLATES.items():
            register_prompt(f"graph/got/verifier:{dataset}", text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("GoT prompt bootstrap skipped: %s", exc)

    try:
        from coscope.graph.cot import prompt_templates as cot_pt

        for name in dir(cot_pt):
            if name.startswith("COT_") and isinstance(getattr(cot_pt, name), str):
                suffix = name[len("COT_"):].lower()
                register_prompt(f"graph/cot/{suffix}", getattr(cot_pt, name))
    except Exception as exc:  # noqa: BLE001
        logger.debug("CoT prompt bootstrap skipped: %s", exc)

    try:
        from coscope.graph.tot import prompt_templates as tot_pt

        for name in dir(tot_pt):
            if name.startswith("TOT_") and isinstance(getattr(tot_pt, name), str):
                suffix = name[len("TOT_"):].lower()
                register_prompt(f"graph/tot/{suffix}", getattr(tot_pt, name))
    except Exception as exc:  # noqa: BLE001
        logger.debug("ToT prompt bootstrap skipped: %s", exc)

    # Rollout step-2 / scratch headers. These are short strings that drive
    # the pre-retrieval reasoning and post-retrieval reflection, respectively.
    register_prompt(
        "rollout/step2/header",
        (
            "Step 2: Pre-retrieval reasoning. Reason silently about WHAT evidence you need\n"
            "before issuing a retrieval query. Respond in English, under 50 words, as a\n"
            "short dash-prefixed list of search targets. No preamble.\n"
        ),
    )
    register_prompt(
        "rollout/scratch/header",
        (
            "Post-retrieval private reasoning. Think step-by-step before committing to a conclusion.\n"
            "Respond in English, under 150 words, as a short numbered list of reasoning steps.\n"
            "No markdown headings, no preamble.\n"
        ),
    )


_bootstrap_builtin_prompts()


__all__ = [
    "PromptBackend",
    "InMemoryBackend",
    "PromptRegistry",
    "register_prompt",
    "get_prompt",
    "has_prompt",
    "set_backend",
    "list_keys",
]
