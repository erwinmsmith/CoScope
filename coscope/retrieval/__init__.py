"""
CoScope Retrieval Pipeline.

Modular retrieval system with the following sub-modules:
- encoder: Request normalization and encoding
- router: Scope-based request bucketing
- matrix: Query matrix construction
- projection: Shared subspace projection
- retriever: Candidate retrieval
- reranker: Agent-specific reranking
- fallback: Private scope fallback
- fusion: Evidence fusion
"""

# Main pipeline and types
from coscope.retrieval.pipeline import PipelineConfig, RetrievalPipeline, create_pipeline

# Encoder module
from coscope.retrieval.encoder import (
    RequestEncoder,
    EncodingResult,
    NormalizationStrategy,
    BasicNormalizer,
    RoleAwareNormalizer,
    EmbeddingAdapter,
)

# Router module
from coscope.retrieval.router import (
    ScopeRouter,
    RoutingResult,
    RetrievalBucket,
    HierarchicalRouter,
    OverlapAwareRouter,
)

# Matrix module
from coscope.retrieval.matrix import QueryMatrixBuilder, QueryMatrix

# Projection module
from coscope.retrieval.projection import SharedProjectionModule, ProjectionConfig

# Retriever module
from coscope.retrieval.retriever import (
    CandidateRetriever,
    SharedCandidateRetriever,
    CosineSimilarity,
    DotProductSimilarity,
)

# Reranker module
from coscope.retrieval.reranker import (
    CandidateReranker,
    WeightedReranker,
    RoleAwareReranker,
)

# Fallback module
from coscope.retrieval.fallback import (
    FallbackTrigger,
    FallbackResult,
    CountBasedTrigger,
    CombinedTrigger,
)

# Fusion module
from coscope.retrieval.fusion import (
    EvidenceFusion,
    FusionResult,
    ReciprocalRankFusion,
    ScoreWeightedFusion,
)

__all__ = [
    # Pipeline
    "RetrievalPipeline",
    "PipelineConfig",
    "create_pipeline",
    # Encoder
    "RequestEncoder",
    "EncodingResult",
    "NormalizationStrategy",
    "BasicNormalizer",
    "RoleAwareNormalizer",
    "EmbeddingAdapter",
    # Router
    "ScopeRouter",
    "RoutingResult",
    "RetrievalBucket",
    "HierarchicalRouter",
    "OverlapAwareRouter",
    # Matrix
    "QueryMatrixBuilder",
    "QueryMatrix",
    # Projection
    "SharedProjectionModule",
    "ProjectionConfig",
    # Retriever
    "CandidateRetriever",
    "SharedCandidateRetriever",
    "CosineSimilarity",
    "DotProductSimilarity",
    # Reranker
    "CandidateReranker",
    "WeightedReranker",
    "RoleAwareReranker",
    # Fallback
    "FallbackTrigger",
    "FallbackResult",
    "CountBasedTrigger",
    "CombinedTrigger",
    # Fusion
    "EvidenceFusion",
    "FusionResult",
    "ReciprocalRankFusion",
    "ScoreWeightedFusion",
]
