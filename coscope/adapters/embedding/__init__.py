from coscope.adapters.embedding.base import EmbeddingAdapter
from coscope.adapters.embedding.dashscope import DashScopeEmbedding
from coscope.adapters.embedding.deterministic import DeterministicEmbedding
from coscope.adapters.embedding.fastembed import FastEmbedEmbedding
from coscope.adapters.embedding.zhipu import ZhipuEmbedding

__all__ = [
    "DashScopeEmbedding",
    "DeterministicEmbedding",
    "EmbeddingAdapter",
    "FastEmbedEmbedding",
    "ZhipuEmbedding",
]
