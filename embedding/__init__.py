"""Embedder implementations. Protocol is defined in `core.interfaces.Embedder`."""

from embedding.dashscope_embedder import DashScopeEmbedder
from embedding.sentence_transformer_embedder import SentenceTransformerEmbedder

__all__ = ["DashScopeEmbedder", "SentenceTransformerEmbedder"]
