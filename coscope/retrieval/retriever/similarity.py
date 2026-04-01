"""
Similarity computation strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class SimilarityStrategy(ABC):
    """Base class for similarity computation strategies."""

    @abstractmethod
    def compute(self, Z: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix.

        Args:
            Z: Query representations (n, r)
            K: Key/candidate representations (m, r)

        Returns:
            Similarity matrix (n, m)
        """
        ...


class CosineSimilarity(SimilarityStrategy):
    """Cosine similarity strategy."""

    def compute(self, Z: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        K_norm = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
        return Z_norm @ K_norm.T


class DotProductSimilarity(SimilarityStrategy):
    """Plain dot product similarity."""

    def compute(self, Z: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Compute dot product."""
        return Z @ K.T


class EuclideanSimilarity(SimilarityStrategy):
    """Euclidean distance converted to similarity."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def compute(self, Z: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Compute negative squared Euclidean distance."""
        Z_sq = np.sum(Z**2, axis=1, keepdims=True)
        K_sq = np.sum(K**2, axis=1, keepdims=True)
        distances_sq = Z_sq + K_sq.T - 2 * (Z @ K.T)
        return -distances_sq / self.temperature
