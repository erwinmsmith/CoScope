"""Retrieval Encoder Module."""

from coscope.retrieval.encoder.base import (
    RequestEncoder,
    EncodingResult,
)
from coscope.retrieval.encoder.normalizer import (
    NormalizationStrategy,
    BasicNormalizer,
    RoleAwareNormalizer,
    CompositeNormalizer,
)
from coscope.retrieval.encoder.adapter import (
    EmbeddingAdapter,
    LangChainEmbeddingAdapter,
)

__all__ = [
    "RequestEncoder",
    "EncodingResult",
    "NormalizationStrategy",
    "BasicNormalizer",
    "RoleAwareNormalizer",
    "CompositeNormalizer",
    "EmbeddingAdapter",
    "LangChainEmbeddingAdapter",
]
