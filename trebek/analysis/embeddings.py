"""
Vector embedding distance calculations for semantic analysis.

Provides cosine distance and lateral semantic distance metrics
used to measure the relationship between J! clue texts
and their correct responses (e.g., wordplay vs. direct recall).
"""

import math
import structlog
from typing import List

logger = structlog.get_logger()


def cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculates the cosine distance between two floating-point vectors.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Embeddings must have the same dimensionality.")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0  # Max distance if one vector is empty

    similarity = dot_product / (norm_a * norm_b)
    return max(0.0, min(1.0, 1.0 - similarity))


def process_semantic_lateral_distance(clue_embedding: List[float], response_embedding: List[float]) -> float:
    """
    Calculates the lateral semantic distance between a clue and the correct response.
    High distance = wordplay/lateral thinking. Low distance = direct factual recall.
    """
    distance = cosine_distance(clue_embedding, response_embedding)
    logger.info("Calculated Semantic Lateral Distance", distance=round(distance, 4))
    return distance
