"""
Prompt Management Module for CoScope.

Separates prompt templates from core logic for easy customization
and externalization.
"""

from coscope.prompts.manager import PromptManager
from coscope.prompts.templates import PromptTemplate, RoleTemplates

__all__ = [
    "PromptManager",
    "PromptTemplate",
    "RoleTemplates",
]
