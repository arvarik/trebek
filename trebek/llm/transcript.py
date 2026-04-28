"""
Transcript formatting and compression utilities for the LLM extraction pipeline.

Provides functions to compress WhisperX transcript segments into a token-efficient
format for Gemini API consumption:
- Timestamps removed (LLM uses Line IDs instead)
- Speaker IDs abbreviated (SPEAKER_00 → S0)
- Bracket noise removed
"""

from typing import Any, Dict


def _abbreviate_speaker(speaker: str) -> str:
    """Compress SPEAKER_XX → SXX for token savings (3 tokens → 1 token per line)."""
    if speaker.startswith("SPEAKER_"):
        return "S" + speaker[len("SPEAKER_") :]
    return speaker


def _build_speaker_abbreviation_map(segments: list[Dict[str, Any]]) -> Dict[str, str]:
    """Build a lookup from abbreviated speaker IDs back to original SPEAKER_XX IDs."""
    abbrev_map: Dict[str, str] = {}
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        abbrev = _abbreviate_speaker(speaker)
        if abbrev != speaker:
            abbrev_map[abbrev] = speaker
    return abbrev_map


def _format_transcript_compressed(segments: list[Dict[str, Any]]) -> str:
    """
    Format transcript lines with prompt compression:
    - Timestamps REMOVED (LLM uses Line IDs, timestamps resolved post-extraction)
    - Speaker IDs abbreviated (SPEAKER_00 → S0)
    - Bracket noise removed ([L0] → L0)

    Before: [L0] [0.00s] SPEAKER_00: Welcome to J!
    After:  L0 S0: Welcome to J!
    """
    formatted_lines = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        speaker = _abbreviate_speaker(seg.get("speaker", "UNKNOWN"))
        formatted_lines.append(f"L{i} {speaker}: {text}")

    return "\n".join(formatted_lines)
