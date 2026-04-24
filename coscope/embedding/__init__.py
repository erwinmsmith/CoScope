"""Embedder implementations. Protocol is defined in `coscope.core.interfaces.Embedder`."""

from coscope.embedding.dashscope_embedder import DashScopeEmbedder
from coscope.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder

__all__ = ["DashScopeEmbedder", "SentenceTransformerEmbedder"]
