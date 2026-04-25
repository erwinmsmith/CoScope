"""
Template-based LLMClient implementation.

Deterministically maps prompts to synthetic responses using heuristics over
the prompt prefix. Used for pipeline validation and CI when a real LLM
backend (Qwen, OpenAI, ...) is not available.

This client is stateless and reproducible: identical prompts produce identical
responses regardless of system state.
"""

from __future__ import annotations

import hashlib
import time
from typing import List, Optional

from core.interfaces import LLMResponse


class TemplateLLMClient:
    """Implements the LLMClient protocol via prompt-prefix dispatch."""

    name: str = "template-llm"
    max_tokens: int = 512

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stop: Optional[List[str]] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> LLMResponse:
        t0 = time.time()
        text = self._dispatch(prompt)
        return LLMResponse(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            finish_reason="stop",
            model=self.name,
            latency_ms=(time.time() - t0) * 1000.0,
            raw={"seed": seed},
        )

    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: Optional[str] = None,
        stop: Optional[List[str]] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[LLMResponse]:
        return [
            self.generate(
                p,
                system=system,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_new_tokens=max_new_tokens,
            )
            for p in prompts
        ]

    # ------------------------------------------------------------------

    @staticmethod
    def _dispatch(prompt: str) -> str:
        """Prompt-prefix heuristic dispatch. Keeps output short and stable."""
        fp = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
        if "Step 2 Pre-Retrieval" in prompt:
            return f"[TEMPLATE query_intent {fp}]\nI need to retrieve the supporting evidence for this step."
        if "Post-retrieval private reasoning" in prompt:
            return f"[TEMPLATE scratch {fp}]\nReasoning over ancestor conclusions and retrieved evidence."
        if "你是一个图状推理规划者" in prompt or "planner" in prompt.lower():
            return f"[TEMPLATE plan {fp}]\nDecompose into sequential solver hops."
        if "你是推理图节点" in prompt or "solver" in prompt.lower():
            return f"[TEMPLATE conclusion {fp}]\nBased on evidence, the intermediate conclusion holds."
        if "核查" in prompt or "verifier" in prompt.lower():
            return f"[TEMPLATE audit_report {fp}]\nAll conclusions reviewed; no provenance issues."
        return f"[TEMPLATE generic {fp}]"
