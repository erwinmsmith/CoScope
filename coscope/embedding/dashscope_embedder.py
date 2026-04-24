"""
DashScope text-embedding-v3 embedder.

Implements the Embedder Protocol defined in `coscope.core.interfaces`.
- Default dim 1024 (v3 supports 1024 / 768 / 512).
- Max 25 texts per API call; this class auto-chunks.
- Reads API key from env `DASHSCOPE_API_KEY`.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import numpy as np
from openai import OpenAI

from coscope.core.interfaces import Embedder, EmbeddingResult


logger = logging.getLogger(__name__)


_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_BATCH_LIMIT = 25          # DashScope hard cap per request
_MAX_CHARS = 8192 * 3      # ~8192 tokens safety truncation


class DashScopeEmbedder(Embedder):
    """text-embedding-v3 via DashScope OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "text-embedding-v3",
        dim: int = 1024,
        api_key: Optional[str] = None,
        base_url: str = _DASHSCOPE_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("DASHSCOPE_API_KEY not set.")
        self._client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        self.model = model
        self.name = f"dashscope:{model}"
        self.dim = int(dim)
        self.max_retries = max_retries

    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        res = self.embed_batch([text])
        return res.vectors[0]

    def embed_batch(self, texts: List[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=np.zeros((0, self.dim), dtype=np.float32),
                                   dim=self.dim, model=self.model)
        safe_texts = [self._truncate(t) for t in texts]
        vecs: List[np.ndarray] = []
        for i in range(0, len(safe_texts), _BATCH_LIMIT):
            chunk = safe_texts[i : i + _BATCH_LIMIT]
            vecs.append(self._call_api(chunk))
        stacked = np.vstack(vecs).astype(np.float32)
        return EmbeddingResult(
            vectors=stacked,
            dim=self.dim,
            model=self.model,
            meta={"count": len(texts)},
        )

    # ------------------------------------------------------------------

    def _call_api(self, texts: List[str]) -> np.ndarray:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=self.dim,
                    encoding_format="float",
                )
                vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
                return vectors
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wait = min(2 ** attempt, 8)
                logger.warning(
                    "DashScope embed attempt %d/%d failed: %s; retry in %ds",
                    attempt + 1, self.max_retries + 1, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"DashScope embed failed after retries: {last_err}")

    @staticmethod
    def _truncate(text: str) -> str:
        if not text:
            return " "
        if len(text) > _MAX_CHARS:
            return text[:_MAX_CHARS]
        return text
