"""
Sentence-Transformers embedder adapter.

Implements both the canonical `coscope.core.interfaces.Embedder` Protocol
(``embed`` / ``embed_batch``) and the retrieval-side `EmbeddingProvider`
shape expected by the engine pipeline (``embed_query`` / ``embed_texts``),
so a single instance can be dropped into either code path.

Typical smoke / offline evaluation model choices:
- ``sentence-transformers/all-MiniLM-L6-v2`` (384-d, ~90MB, English, fast)
- ``BAAI/bge-small-en-v1.5`` (384-d, stronger English retrieval)
- ``BAAI/bge-m3`` (1024-d, multilingual; heavier download)

Downloads are cached under HuggingFace's default `~/.cache/huggingface`.
No network calls happen after the first load.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from coscope.core.interfaces import Embedder, EmbeddingResult


logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(Embedder):
    """Offline sentence-transformers adapter."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 64,
    ):
        """
        Parameters
        ----------
        model_name:
            Any identifier accepted by ``SentenceTransformer``. Local paths
            are also supported.
        device:
            Override the inference device ("cpu", "cuda", "cuda:0", ...).
            ``None`` lets sentence-transformers pick automatically.
        normalize:
            Whether to L2-normalize the output vectors. Cosine-based retrieval
            downstream assumes normalized inputs.
        batch_size:
            Batch size for ``embed_batch`` / ``embed_texts``.
        """
        # Local import so CoScope can be imported without the optional dep.
        from sentence_transformers import SentenceTransformer

        self._st = SentenceTransformer(model_name, device=device)
        self.name = f"st:{model_name}"
        self.normalize = normalize
        self.batch_size = batch_size

        # Determine dim from a dry-run; cheaper than encoding a real string
        # multiple times downstream.
        self.dim = int(self._st.get_sentence_embedding_dimension())
        logger.info(
            "Loaded SentenceTransformerEmbedder(%s, dim=%d, device=%s, normalize=%s)",
            model_name, self.dim, device or "auto", normalize,
        )

    # ------------------------------------------------------------------
    # coscope.core.interfaces.Embedder (canonical)
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        vec = self._st.encode(
            [text],
            batch_size=1,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vec.astype("float32")

    def embed_batch(self, texts: List[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(
                vectors=np.zeros((0, self.dim), dtype="float32"),
                dim=self.dim,
                model=self.name,
            )
        vectors = self._st.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        return EmbeddingResult(
            vectors=vectors,
            dim=self.dim,
            model=self.name,
            meta={"n": len(texts)},
        )

    # ------------------------------------------------------------------
    # engine.EmbeddingProvider (retrieval pipeline shape)
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed(query)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        result = self.embed_batch(texts)
        return [row for row in result.vectors]


__all__ = ["SentenceTransformerEmbedder"]
