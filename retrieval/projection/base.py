"""
Shared Projection Module - Projects query matrices to shared subspace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from retrieval.matrix.types import QueryMatrix

logger = logging.getLogger(__name__)


@dataclass
class ProjectionConfig:
    """Configuration for the projection module."""

    input_dim: int = 1024  # k
    intermediate_dim: int = 256  # l
    svd_rank: int = 64  # r
    use_mask: bool = True
    mask_sparsity: float = 0.5
    initializer: str = "xavier"


@dataclass
class ProjectionResult:
    """Result of projection transformation."""

    projected: np.ndarray  # Z ∈ R^(n × r)
    projection_matrix: np.ndarray  # W_final ∈ R^(k × r)
    svd_rank: int
    mask_applied: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class SharedProjectionModule:
    """
    Performs projection transformation on query matrices.

    Computes:
    1. W' = W ⊙ M (masked projection)
    2. SVD: W' = U Σ V^T
    3. W_final = V[:, 1:r] (shared subspace)
    4. Z = Q^T @ W_final (projected queries)
    """

    def __init__(
        self,
        config: Optional[ProjectionConfig] = None,
    ):
        self.config = config or ProjectionConfig()
        self._initialized = False
        self._W_final: Optional[np.ndarray] = None
        self._W_masked: Optional[np.ndarray] = None
        self._singular_values: Optional[np.ndarray] = None

    def fit(
        self,
        query_matrices: List["QueryMatrix"],
    ) -> ProjectionResult:
        """
        Fit the projection module on query matrices.

        Args:
            query_matrices: List of query matrices to fit on

        Returns:
            The computed ProjectionResult
        """
        if not query_matrices:
            raise ValueError("At least one query matrix required for fitting")

        k = self.config.input_dim
        l = self.config.intermediate_dim

        # Initialize projection matrix
        W = self._initialize_projection_matrix(k, l)

        # Apply mask
        if self.config.use_mask:
            M = self._create_sparse_mask(l, k)
            W_masked = W * M
        else:
            M = np.ones_like(W)
            W_masked = W

        # SVD
        try:
            U, S, VT = np.linalg.svd(W_masked, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback for degenerate case
            W_masked = W_masked + np.random.randn(*W_masked.shape).astype(np.float32) * 1e-6
            U, S, VT = np.linalg.svd(W_masked, full_matrices=False)

        # Extract shared subspace
        r = min(self.config.svd_rank, len(S))
        W_final = VT[:r, :].T  # shape (k, r)

        # Store
        self._W_final = W_final
        self._W_masked = W_masked
        self._singular_values = S
        self._initialized = True

        logger.info(f"Fitted projection: k={k}, l={l}, r={r}")

        return ProjectionResult(
            projected=np.array([]),  # Not computed during fit
            projection_matrix=W_final,
            svd_rank=r,
            mask_applied=self.config.use_mask,
            metadata={
                "singular_values_top5": S[:5].tolist() if len(S) >= 5 else S.tolist(),
                "mask_sparsity": float(np.sum(M == 0) / M.size) if self.config.use_mask else 0,
            },
        )

    def transform(
        self,
        query_matrix: "QueryMatrix",
    ) -> ProjectionResult:
        """
        Transform a query matrix to the shared subspace.

        Z = Q^T @ W_final ∈ R^(n × r)

        Args:
            query_matrix: Input query matrix Q ∈ R^(k × n)

        Returns:
            ProjectionResult containing Z and metadata
        """
        if not self._initialized:
            raise RuntimeError("Projection module not fitted. Call fit() first.")

        Q = query_matrix.embeddings

        if Q.shape[0] != self._W_final.shape[0]:
            raise ValueError(
                f"Query embedding dim {Q.shape[0]} != projection input dim {self._W_final.shape[0]}"
            )

        # Compute Z = Q^T @ W_final
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Z = Q.T @ self._W_final
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

        return ProjectionResult(
            projected=Z,
            projection_matrix=self._W_final,
            svd_rank=self.config.svd_rank,
            mask_applied=self.config.use_mask,
            metadata={
                "Z_shape": Z.shape,
                "W_final_shape": self._W_final.shape,
            },
        )

    def fit_transform(
        self,
        query_matrices: List["QueryMatrix"],
    ) -> List[ProjectionResult]:
        """Fit and transform in one step."""
        self.fit(query_matrices)
        return [self.transform(qm) for qm in query_matrices]

    def _initialize_projection_matrix(self, k: int, l: int) -> np.ndarray:
        """Initialize projection matrix."""
        init_type = self.config.initializer

        if init_type == "xavier":
            limit = np.sqrt(6.0 / (k + l))
            W = np.random.uniform(-limit, limit, (l, k)).astype(np.float32)
        elif init_type == "random":
            W = np.random.randn(l, k).astype(np.float32)
            W = W / (np.sum(W**2, axis=1, keepdims=True) ** 0.5 + 1e-8)
        else:
            W = np.random.randn(l, k).astype(np.float32) * 0.01

        return W

    def _create_sparse_mask(self, l: int, k: int) -> np.ndarray:
        """Create structured sparse mask."""
        sparsity = self.config.mask_sparsity
        mask = np.random.rand(l, k)
        return (mask > sparsity).astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._initialized

    def get_stats(self) -> Dict[str, Any]:
        """Get projection statistics."""
        if not self._initialized:
            return {"initialized": False}

        return {
            "initialized": True,
            "input_dim": self.config.input_dim,
            "intermediate_dim": self.config.intermediate_dim,
            "svd_rank": self.config.svd_rank,
            "mask_applied": self.config.use_mask,
            "singular_values_top5": (
                self._singular_values[:5].tolist()
                if self._singular_values is not None
                else None
            ),
        }
