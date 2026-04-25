"""
Prompt templates for retrieval and agent interactions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A prompt template with variable substitution."""

    name: str
    template: str
    description: str = ""
    variables: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return self.template

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            template=data["template"],
            description=data.get("description", ""),
            variables=data.get("variables", []),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "description": self.description,
            "variables": self.variables,
            "metadata": self.metadata,
        }


class RoleTemplates:
    """
    Pre-defined prompt templates for different agent roles.
    """

    PLANNER = PromptTemplate(
        name="planner_retrieval",
        template="Planning: {query}\nContext: {task_context}",
        description="Template for planner agent retrieval queries",
        variables=["query", "task_context"],
    )

    SOLVER = PromptTemplate(
        name="solver_retrieval",
        template="Solving: {query}\nCurrent step: {plan_node}",
        description="Template for solver agent retrieval queries",
        variables=["query", "plan_node"],
    )

    VERIFIER = PromptTemplate(
        name="verifier_retrieval",
        template="Verify: {query}\nEvidence needed for: {evidence_type}",
        description="Template for verifier agent retrieval queries",
        variables=["query", "evidence_type"],
    )

    CRITIC = PromptTemplate(
        name="critic_retrieval",
        template="Critique: {query}\nLooking for conflicts in: {focus_area}",
        description="Template for critic agent retrieval queries",
        variables=["query", "focus_area"],
    )

    MEMORY_MANAGER = PromptTemplate(
        name="memory_manager_retrieval",
        template="Memory: {query}\nScope: {scope}",
        description="Template for memory manager retrieval queries",
        variables=["query", "scope"],
    )

    @classmethod
    def get_template(cls, role: str) -> PromptTemplate:
        """Get template for a role."""
        templates = {
            "planner": cls.PLANNER,
            "solver": cls.SOLVER,
            "verifier": cls.VERIFIER,
            "critic": cls.CRITIC,
            "memory_manager": cls.MEMORY_MANAGER,
        }
        return templates.get(role.lower(), cls.PLANNER)

    @classmethod
    def all_templates(cls) -> Dict[str, PromptTemplate]:
        """Get all templates as a dictionary."""
        return {
            "planner": cls.PLANNER,
            "solver": cls.SOLVER,
            "verifier": cls.VERIFIER,
            "critic": cls.CRITIC,
            "memory_manager": cls.MEMORY_MANAGER,
        }

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert all templates to dictionary."""
        return {name: tpl.to_dict() for name, tpl in cls.all_templates().items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> None:
        """Load templates from dictionary (class method for loading)."""
        for name, tpl_data in data.items():
            tpl = PromptTemplate.from_dict(tpl_data)
            setattr(cls, name.upper(), tpl)
