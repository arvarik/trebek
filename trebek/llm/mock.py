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


def _build_mock_clues(
    round_name: str,
    categories: list[str],
    contestants: Optional[list[str]] = None,
    base_line_offset: int = 0,
) -> list[dict[str, Any]]:
    """Builds mock clues for a given round with monotonically non-overlapping line ranges."""
    clues: list[dict[str, Any]] = []
    active_contestants = contestants or ["Amy Schneider", "Matt Amodio", "Mattea Roach"]

    for col_idx, category in enumerate(categories, start=1):
        for row_idx in range(1, 6):
            is_dd = (round_name == "J!" and col_idx == 2 and row_idx == 3) or (
                round_name == "Double J!" and ((col_idx == 1 and row_idx == 4) or (col_idx == 4 and row_idx == 3))
            )
            speaker = active_contestants[(col_idx + row_idx) % len(active_contestants)]
            line_base = base_line_offset + (col_idx - 1) * 10 + (row_idx - 1) * 2
            line_start = f"L{line_base}"
            line_end = f"L{line_base + 1}"
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
        import typing

        # Check round from prompt or context
        if "double" in ctx or "double" in prompt.lower():
            round_name = "Double J!"
            default_cats = MOCK_DJ_CATEGORIES
            base_offset = 60
        else:
            round_name = "J!"
            default_cats = MOCK_JEP_CATEGORIES
            base_offset = 0

        target_cats = default_cats
        target_contestants = ["Amy Schneider", "Matt Amodio", "Mattea Roach"]

        # Introspect response_schema if dynamic
        if response_schema and hasattr(response_schema, "model_fields"):
            clue_field = response_schema.model_fields.get("clues")
            if clue_field and typing.get_args(clue_field.annotation):
                clue_type = typing.get_args(clue_field.annotation)[0]
                if hasattr(clue_type, "model_fields"):
                    cat_field = clue_type.model_fields.get("category")
                    if cat_field:
                        allowed = typing.get_args(cat_field.annotation)
                        if allowed:
                            target_cats = list(allowed)
                    attempt_field = clue_type.model_fields.get("attempts")
                    if attempt_field and typing.get_args(attempt_field.annotation):
                        att_type = typing.get_args(attempt_field.annotation)[0]
                        if hasattr(att_type, "model_fields"):
                            spk_field = att_type.model_fields.get("speaker")
                            if spk_field:
                                allowed_spk = typing.get_args(spk_field.annotation)
                                if allowed_spk:
                                    target_contestants = list(allowed_spk)

        # If categories are specified in prompt, filter them
        filtered_cats: list[str] = [c for c in target_cats if c.lower() in prompt.lower()]
        if not filtered_cats:
            filtered_cats = target_cats[: min(3, len(target_cats))]

        clues = _build_mock_clues(
            round_name,
            filtered_cats,
            contestants=target_contestants,
            base_line_offset=base_offset,
        )
        payload = {"clues": clues}

    # 5. Stage 3.5: Verification Results
    elif "batchverificationresult" in schema_name.lower() or "verification" in ctx:
        payload = {
            "verifications": [
                {
                    "clue_index": i - 1,
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
        payload = {
            "verified_correct_response": "What is the Statue of Liberty?",
            "confidence": "verified",
            "correction_type": "",
            "correction_detail": "",
        }

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
