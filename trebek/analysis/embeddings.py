import math
import struct
import structlog
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from trebek.schemas import Clue

logger = structlog.get_logger()


def serialize_embedding(vec: Optional[List[float]]) -> Optional[bytes]:
    """Packs a float vector into a binary blob for SQLite BLOB storage."""
    if not vec:
        return None
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_embedding(blob: Optional[bytes]) -> Optional[List[float]]:
    """Unpacks a binary blob from SQLite back into a float list."""
    if not blob or len(blob) % 4 != 0:
        return None
    dim = len(blob) // 4
    return list(struct.unpack(f"{dim}f", blob))


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


async def enrich_clues_with_embeddings(clues: List["Clue"], client: Any = None) -> None:
    """Generates vector embeddings for clues and responses, computing semantic lateral distance.

    Updates the Clue objects in-place with clue_embedding, response_embedding, and semantic_lateral_distance.
    """
    if not clues:
        return

    if client is None:
        from trebek.llm.client import _get_client

        llm_client: Any = _get_client()
    else:
        llm_client = client

    clue_texts = [c.clue_text for c in clues]
    response_texts = [c.correct_response for c in clues]

    all_texts = clue_texts + response_texts
    try:
        embeddings = await getattr(llm_client, "embed_content")(all_texts)
        n = len(clues)
        clue_embs = embeddings[:n]
        resp_embs = embeddings[n:]

        for i, clue in enumerate(clues):
            c_emb = clue_embs[i] if i < len(clue_embs) else None
            r_emb = resp_embs[i] if i < len(resp_embs) else None

            is_valid_c = c_emb is not None and any(v != 0.0 for v in c_emb)
            is_valid_r = r_emb is not None and any(v != 0.0 for v in r_emb)

            if is_valid_c and is_valid_r and c_emb is not None and r_emb is not None:
                clue.clue_embedding = c_emb
                clue.response_embedding = r_emb
                try:
                    clue.semantic_lateral_distance = process_semantic_lateral_distance(c_emb, r_emb)
                except Exception as dist_err:
                    logger.debug(
                        "Failed to calculate semantic lateral distance",
                        clue=clue.clue_text[:30],
                        error=str(dist_err),
                    )
                    clue.semantic_lateral_distance = None
            else:
                clue.clue_embedding = c_emb if is_valid_c else None
                clue.response_embedding = r_emb if is_valid_r else None
                clue.semantic_lateral_distance = None
    except Exception as e:
        logger.warning("Failed to generate embeddings for clues", count=len(clues), error=str(e)[:200])
