"""
Prompt Management Module for CoScope.

Separates prompt templates from core logic for easy customization
and externalization.
"""

from coscope.prompts.manager import PromptManager
from coscope.prompts.templates import PromptTemplate, RoleTemplates
from coscope.prompts.registry import (
    PromptBackend,
    InMemoryBackend,
    PromptRegistry,
    get_prompt,
    has_prompt,
    list_keys,
    register_prompt,
    set_backend,
)

__all__ = [
    "PromptManager",
    "PromptTemplate",
    "RoleTemplates",
    # Central registry (DB-pluggable)
    "PromptBackend",
    "InMemoryBackend",
    "PromptRegistry",
    "get_prompt",
    "has_prompt",
    "list_keys",
    "register_prompt",
    "set_backend",
]
