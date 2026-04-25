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
from retrieval.pipeline import PipelineConfig, RetrievalPipeline, create_pipeline

# Encoder module
from retrieval.encoder import (
    RequestEncoder,
    EncodingResult,
    NormalizationStrategy,
    BasicNormalizer,
    RoleAwareNormalizer,
    EmbeddingAdapter,
)

# Router module
from retrieval.router import (
    ScopeRouter,
    RoutingResult,
    RetrievalBucket,
    HierarchicalRouter,
    OverlapAwareRouter,
)

# Matrix module
from retrieval.matrix import QueryMatrixBuilder, QueryMatrix

# Projection module
from retrieval.projection import SharedProjectionModule, ProjectionConfig

# Retriever module
from retrieval.retriever import (
    CandidateRetriever,
    SharedCandidateRetriever,
    CosineSimilarity,
    DotProductSimilarity,
)

# Reranker module
from retrieval.reranker import (
    CandidateReranker,
    WeightedReranker,
    RoleAwareReranker,
)

# Fallback module
from retrieval.fallback import (
    FallbackTrigger,
    FallbackResult,
    CountBasedTrigger,
    CombinedTrigger,
)

# Fusion module
from retrieval.fusion import (
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
