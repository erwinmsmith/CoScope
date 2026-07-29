"""Safe construction of public/private retrieval intents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannedIntent:
    full_query: str
    public_intent: str
    private_intent: str | None


class QueryPlanner:
    """The caller must explicitly identify share-safe public text.

    The runtime deliberately avoids heuristic redaction: silently guessing that
    a full query is public could encode private state into a shared vector.
    """

    def plan(
        self,
        full_query: str,
        *,
        public_intent: str | None = None,
        private_intent: str | None = None,
        full_query_is_public: bool = False,
    ) -> PlannedIntent:
        if not full_query.strip():
            raise ValueError("full_query is required")
        public = public_intent.strip() if public_intent else ""
        if not public and full_query_is_public:
            public = full_query.strip()
        if not public:
            raise ValueError("public_intent must be supplied or explicitly marked public")
        private = private_intent.strip() if private_intent else None
        if private and private in public:
            raise ValueError("public_intent contains the declared private intent")
        return PlannedIntent(full_query.strip(), public, private)
