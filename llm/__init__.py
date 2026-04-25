"""
CoScope LLM layer.

Owns the :class:`LLMClient` protocol plus the concrete clients
(:class:`DashScopeClient`, :class:`TemplateLLMClient`). Other subsystems
(rollout engine, scripts, examples) depend on this layer rather than owning
LLM access themselves. Plug additional providers in by implementing the
:class:`LLMClient` protocol and exporting them here.
"""

from core.interfaces import LLMClient, LLMResponse
from llm.dashscope import DashScopeClient
from llm.template import TemplateLLMClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "DashScopeClient",
    "TemplateLLMClient",
]
