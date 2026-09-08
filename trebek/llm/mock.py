"""
Synthetic Mock LLM Engine for zero-cost offline local development and smoke testing.

Generates schema-compliant, deterministic, and realistic Jeopardy! game responses
for Pass 1 (speaker anchoring), Pass 2 (skeleton, metadata, and clue extraction),
and Pass 3 (multimodal augmentation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MockCandidate:
    finish_reason: str = "STOP"


@dataclass
class MockResponse:
    text: str
    candidates: list[MockCandidate] = field(default_factory=lambda: [MockCandidate()])


MOCK_JEP_CATEGORIES = [
    "WORLD CAPITALS",
    "AMERICAN HISTORY",
    "LITERATURE",
    "SCIENCE & NATURE",
    "POP MUSIC",
    "WORD ORIGINS",
]

MOCK_DJ_CATEGORIES = [
    "EUROPEAN RULERS",
    "PHYSICS",
    "AUTHORS",
    "ISLANDS",
    "OPERA",
    "BEFORE & AFTER",
]

MOCK_CONTESTANTS = [
    {
        "podium_position": 1,
        "name": "Amy Schneider",
        "occupational_category": "STEM",
        "is_returning_champion": True,
        "description": "Software engineer from Oakland, California.",
    },
    {
        "podium_position": 2,
        "name": "Matt Amodio",
        "occupational_category": "Academia",
        "is_returning_champion": False,
        "description": "Ph.D. student from New Haven, Connecticut.",
    },
    {
        "podium_position": 3,
        "name": "Mattea Roach",
        "occupational_category": "Law",
        "is_returning_champion": False,
        "description": "Tutor from Toronto, Ontario, Canada.",
    },
]


def _build_mock_clues(round_name: str, categories: list[str]) -> list[dict[str, Any]]:
    """Builds a full 30-clue board for a given round."""
    clues: list[dict[str, Any]] = []
    contestants = ["Amy Schneider", "Matt Amodio", "Mattea Roach"]

    for col_idx, category in enumerate(categories, start=1):
        for row_idx in range(1, 6):
            is_dd = (round_name == "J!" and col_idx == 2 and row_idx == 3) or (
                round_name == "Double J!" and ((col_idx == 1 and row_idx == 4) or (col_idx == 4 and row_idx == 3))
            )
            speaker = contestants[(col_idx + row_idx) % 3]
            line_start = f"L{(col_idx - 1) * 10 + row_idx * 2}"
            line_end = f"L{(col_idx - 1) * 10 + row_idx * 2 + 1}"
            buzz_line = line_end

            clue = {
                "round": round_name,
                "category": category,
                "board_row": row_idx,
                "board_col": col_idx,
                "is_daily_double": is_dd,
                "requires_visual_context": False,
                "host_read_start_line_id": line_start,
                "host_read_end_line_id": line_end,
                "daily_double_wager": "2000" if is_dd else None,
                "wagerer_name": speaker if is_dd else None,
                "clue_text": f"This clue from {category} tests classic trivia knowledge for row {row_idx}.",
                "correct_response": f"What is answer for {category} {row_idx}?",
                "is_verified": False,
                "original_response": None,
                "attempts": [
                    {
                        "attempt_order": 1,
                        "speaker": speaker,
                        "response_given": f"What is answer for {category} {row_idx}?",
                        "is_correct": True,
                        "buzz_line_id": buzz_line,
                        "is_lockout_inferred": False,
                    }
                ],
            }
            clues.append(clue)
    return clues


def generate_mock_llm_response(
    model: str,
    prompt: str,
    response_schema: Optional[Any] = None,
    invocation_context: str = "",
) -> tuple[MockResponse, dict[str, float]]:
    """
    Generates a deterministic synthetic response matching the requested schema or context.
    """
    ctx = invocation_context.lower()
    schema_name = getattr(response_schema, "__name__", "") if response_schema else ""

    payload: dict[str, Any]

    # 1. Pass 1: Speaker Diarization Mapping
    if "diarizationmapping" in schema_name.lower() or "pass 1" in ctx or "speaker anchoring" in ctx:
        payload = {
            "SPEAKER_00": "Ken Jennings",
            "SPEAKER_01": "Amy Schneider",
            "SPEAKER_02": "Matt Amodio",
            "SPEAKER_03": "Mattea Roach",
        }

    # 2. Pass 2 Stage 1: Episode Skeleton
    elif "episodeskeleton" in schema_name.lower() or "skeleton" in ctx:
        payload = {
            "jeopardy_categories": MOCK_JEP_CATEGORIES,
            "double_jep_categories": MOCK_DJ_CATEGORIES,
            "total_jep_clues_played": 30,
            "total_double_jep_clues_played": 30,
            "daily_double_count": 3,
        }

    # 3. Pass 2 Stage 2: Partial Episode Meta
    elif "partialepisodemeta" in schema_name.lower() or "metadata" in ctx:
        payload = {
            "episode_date": "2024-05-15",
            "host_name": "Ken Jennings",
            "is_tournament": False,
            "contestants": MOCK_CONTESTANTS,
            "jeopardy_categories": MOCK_JEP_CATEGORIES,
            "double_jep_categories": MOCK_DJ_CATEGORIES,
            "final_jep": {
                "category": "HISTORIC MONUMENTS",
                "clue_text": "Dedicated in 1886, this copper colossus was designed by Frédéric-Auguste Bartholdi.",
                "correct_response": "What is the Statue of Liberty?",
                "wagers_and_responses": [
                    {
                        "contestant": "Amy Schneider",
                        "wager": 6000,
                        "response": "What is the Statue of Liberty?",
                        "is_correct": True,
                    },
                    {
                        "contestant": "Matt Amodio",
                        "wager": 4000,
                        "response": "What is the Statue of Liberty?",
                        "is_correct": True,
                    },
                    {
                        "contestant": "Mattea Roach",
                        "wager": 4000,
                        "response": "What is the Statue of Liberty?",
                        "is_correct": True,
                    },
                ],
            },
            "score_adjustments": [],
        }

    # 4. Pass 2 Stage 3: Clue Extractions (PartialClues or DynamicPartialClues)
    elif "partialclues" in schema_name.lower() or "clue" in ctx:
        # Check round from prompt or context
        if "double" in ctx or "double" in prompt.lower():
            round_name = "Double J!"
            cats = MOCK_DJ_CATEGORIES
        else:
            round_name = "J!"
            cats = MOCK_JEP_CATEGORIES

        # If categories are specified in prompt, extract or adapt them
        filtered_cats: list[str] = []
        for c in cats:
            if c.lower() in prompt.lower():
                filtered_cats.append(c)
        if not filtered_cats:
            filtered_cats = cats[:3]  # Chunk typically has 2-3 categories

        clues = _build_mock_clues(round_name, filtered_cats)
        payload = {"clues": clues}

    # 5. Stage 3.5: Verification Results
    elif "batchverificationresult" in schema_name.lower() or "verification" in ctx:
        payload = {
            "verifications": [
                {
                    "selection_order": i,
                    "verified_clue_text": "Sample verified clue text.",
                    "verified_correct_response": "What is verified answer?",
                    "confidence": "verified",
                    "correction_type": "",
                    "correction_detail": "",
                }
                for i in range(1, 13)
            ]
        }

    elif "fjverification" in schema_name.lower() or "final" in ctx:
        payload = {"verified_correct_response": "What is the Statue of Liberty?"}

    # Default generic payload
    else:
        payload = {
            "mock": True,
            "status": "success",
            "message": f"Synthetic response for context '{invocation_context}'",
        }

    json_str = json.dumps(payload)
    usage = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "thinking_tokens": 0.0,
        "cached_tokens": 0.0,
        "total_tokens": 0.0,
        "cost_usd": 0.0,
        "latency_ms": 1.0,
        "mock": 1.0,
    }

    return MockResponse(text=json_str), usage


def generate_mock_embedding(text: str, dim: int = 768) -> list[float]:
    """Generates a deterministic, unit-normalized 768-dimensional float vector.

    Derived from SHA-256 hash of the input text, providing reproducible
    vectors with valid cosine similarity behavior for offline mock testing.
    """
    import hashlib
    import math

    if not text:
        return [0.0] * dim

    # Use text hash as seed material
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed_int = int.from_bytes(h[:8], "big")
    raw_vals: list[float] = []
    state = seed_int
    for _ in range(dim):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        raw_vals.append((state / 0xFFFFFFFFFFFFFFFF) * 2.0 - 1.0)

    norm = math.sqrt(sum(x * x for x in raw_vals))
    if norm == 0.0:
        return [0.0] * dim
    return [round(x / norm, 6) for x in raw_vals]
