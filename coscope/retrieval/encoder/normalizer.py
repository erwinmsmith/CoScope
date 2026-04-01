"""
Query normalization strategies for retrieval requests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from coscope.core.types import AgentRole, AgentState

logger = __import__("logging").getLogger(__name__)


class NormalizationStrategy(ABC):
    """Base class for query normalization strategies."""

    @abstractmethod
    def normalize(
        self,
        query: str,
        role: "AgentRole",
        state: "AgentState",
        metadata: Dict[str, Any],
    ) -> str:
        """
        Normalize a query string based on agent context.

        This may include:
        - Cleaning formatting
        - Adding role context
        - Appending task information
        - Expanding abbreviations
        """
        ...

    def _basic_clean(self, query: str) -> str:
        """Basic text cleaning."""
        query = query.strip()
        query = " ".join(query.split())
        return query


class BasicNormalizer(NormalizationStrategy):
    """Basic normalization without additional context."""

    def normalize(
        self,
        query: str,
        role: "AgentRole",
        state: "AgentState",
        metadata: Dict[str, Any],
    ) -> str:
        return self._basic_clean(query)


class RoleAwareNormalizer(NormalizationStrategy):
    """
    Role-aware normalization that adds relevant context based on agent role.
    """

    # Role-specific prefixes
    ROLE_PREFIXES = {
        "planner": "Planning",
        "solver": "Solving",
        "verifier": "Verify",
        "critic": "Critique",
        "memory_manager": "Memory",
    }

    def __init__(self, include_state: bool = True):
        self.include_state = include_state

    def normalize(
        self,
        query: str,
        role: "AgentRole",
        state: "AgentState",
        metadata: Dict[str, Any],
    ) -> str:
        query = self._basic_clean(query)

        # Add role-specific prefix
        prefix = self.ROLE_PREFIXES.get(role.value, "")
        if prefix and not query.lower().startswith(prefix.lower()):
            query = f"{prefix}: {query}"

        # Add state context if enabled
        if self.include_state and state.task_context:
            task_desc = state.task_context.get("description", "")
            if task_desc:
                query = f"[Task: {task_desc}] {query}"

        return query


class CompositeNormalizer(NormalizationStrategy):
    """
    Composite normalization that chains multiple strategies.
    """

    def __init__(self, strategies: List[NormalizationStrategy]):
        self.strategies = strategies

    def normalize(
        self,
        query: str,
        role: "AgentRole",
        state: "AgentState",
        metadata: Dict[str, Any],
    ) -> str:
        result = query
        for strategy in self.strategies:
            result = strategy.normalize(result, role, state, metadata)
        return result


class TemplateNormalizer(NormalizationStrategy):
    """
    Template-based normalization using prompt templates.
    """

    def __init__(
        self,
        templates: Dict[str, str],
        default_template: str = "{query}",
    ):
        """
        Args:
            templates: Dict mapping role names to template strings
            default_template: Default template if role not found
        """
        self.templates = templates
        self.default_template = default_template

    def normalize(
        self,
        query: str,
        role: "AgentRole",
        state: "AgentState",
        metadata: Dict[str, Any],
    ) -> str:
        template = self.templates.get(role.value, self.default_template)

        # Build context
        context = {
            "query": query,
            "role": role.value,
            "plan_node": state.plan_node or "",
            "task_context": str(state.task_context or {}),
        }

        try:
            return template.format(**context)
        except KeyError:
            return query
