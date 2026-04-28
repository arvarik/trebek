"""
Tests for cosine distance and semantic lateral distance calculations.
Pure math functions, no external dependencies.
"""

import pytest

from trebek.analysis.embeddings import cosine_distance, process_semantic_lateral_distance


class TestCosineDistance:
    """Cosine distance metric tests."""

    def test_identical_vectors_zero_distance(self) -> None:
        vec = [1.0, 2.0, 3.0]
        assert cosine_distance(vec, vec) == 0.0

    def test_orthogonal_vectors_max_distance(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_distance(a, b) == 1.0

    def test_opposite_vectors_max_distance(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_distance(a, b) == 1.0  # Clamped to [0, 1]

    def test_similar_vectors_low_distance(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.1]
        assert cosine_distance(a, b) < 0.01

    def test_zero_vector_returns_max_distance(self) -> None:
        assert cosine_distance([0.0, 0.0], [1.0, 2.0]) == 1.0
        assert cosine_distance([1.0, 2.0], [0.0, 0.0]) == 1.0
        assert cosine_distance([0.0, 0.0], [0.0, 0.0]) == 1.0

    def test_different_dimensionality_raises(self) -> None:
        with pytest.raises(ValueError, match="same dimensionality"):
            cosine_distance([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_single_dimension_identical(self) -> None:
        assert cosine_distance([5.0], [5.0]) == 0.0

    def test_single_dimension_opposite(self) -> None:
        assert cosine_distance([5.0], [-5.0]) == 1.0

    def test_large_vectors(self) -> None:
        a = [float(i) for i in range(100)]
        b = [float(i + 0.01) for i in range(100)]
        dist = cosine_distance(a, b)
        assert 0.0 <= dist < 0.001


class TestSemanticLateralDistance:
    """Integration test for process_semantic_lateral_distance."""

    def test_returns_float(self) -> None:
        result = process_semantic_lateral_distance([1.0, 0.0], [0.0, 1.0])
        assert isinstance(result, float)

    def test_identical_embeddings_zero(self) -> None:
        vec = [1.0, 2.0, 3.0]
        assert process_semantic_lateral_distance(vec, vec) == 0.0

    def test_orthogonal_embeddings_max(self) -> None:
        assert process_semantic_lateral_distance([1.0, 0.0], [0.0, 1.0]) == 1.0
