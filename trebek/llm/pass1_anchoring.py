import ast
import json
import structlog
from typing import Any, Dict, Optional

from trebek.config import MODEL_FLASH
from trebek.llm.client import _get_client

logger = structlog.get_logger()


async def execute_pass_1_speaker_anchoring(audio_file_path: str) -> "tuple[Dict[str, str], dict[str, float]]":
    """
    Pass 1: Fast Gemini 3.1 Flash extraction isolating the host interview.
    Generates a rigid speaker mapping dictionary via Native Audio Anchoring.
    """

    system_prompt = (
        "You are a strict data extractor. Analyze the audio slice of the Jeopardy host interview segment. "
        "Listen to the host interview the contestants. Map their distinct vocal timbres to Diarization Speaker IDs (like SPEAKER_00). "
        "Return a pure JSON dictionary mapping Diarization Speaker IDs to Contestant/Host Names. "
        'Format: {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Matt Amodio"}. '
        "Use double quotes only. Return ONLY the JSON object, no markdown."
    )

    try:
        client = _get_client()
        uploaded_file = await client.upload_file(audio_file_path)

        prompt = [
            "Listen to the host interview the contestants. Map their distinct vocal timbres to the generic SPEAKER_XX IDs.",
            uploaded_file,
        ]

        response, usage = await client.generate_content(
            model=MODEL_FLASH,
            prompt=prompt,
            system_instruction=system_prompt,
            max_output_tokens=65536,
            invocation_context="Pass 1 Speaker Anchoring",
        )

        await client.delete_file(uploaded_file.name)

        # Clean markdown fences if present
        response_text = str(response.text)
        clean = response_text.replace("```json", "").replace("```", "").strip()
        if not clean:
            logger.warning(
                "Pass 1 returned empty response, using empty mapping",
                raw_response=response_text[:200] if response_text else "None",
                input_tokens=int(usage.get("input_tokens", 0)),
                cost_usd=round(usage.get("cost_usd", 0.0), 6),
            )
            return {}, usage

        mapping = _normalize_speaker_mapping(clean)

        logger.info(
            "Pass 1 Speaker Anchor resolved",
            mapping=mapping,
            speakers_found=len(mapping),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cost_usd=round(usage.get("cost_usd", 0.0), 6),
            latency_ms=round(usage.get("latency_ms", 0), 0),
        )
        return mapping, usage
    except Exception as e:
        logger.error("Failed to generate speaker anchor", error=str(e), error_type=type(e).__name__)
        # Fallback or default mapping logic here
        return {}, {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "thinking_tokens": 0.0,
            "cached_tokens": 0.0,
            "total_tokens": 0.0,
            "cost_usd": 0.0,
            "latency_ms": 0.0,
        }


def _normalize_speaker_mapping(raw_text: str) -> Dict[str, str]:
    """
    Robustly extracts SPEAKER_XX → name pairs from any format the LLM returns.

    Handles:
    - Standard dict: {"SPEAKER_00": "Ken Jennings", ...}
    - Single-quoted Python dict: {'SPEAKER_00': 'Ken'}
    - List of dicts: [{"speaker": "SPEAKER_00", "name": "Ken"}, ...]
    - Nested dict: {"speakers": {"SPEAKER_00": "Ken"}}
    - Prose with patterns: "SPEAKER_00 is Ken Jennings"
    """
    import re

    # Strategy 1: Try json.loads
    try:
        parsed = json.loads(raw_text)
        result = _extract_from_parsed(parsed)
        if result:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Try ast.literal_eval (handles single-quoted dicts)
    try:
        parsed = ast.literal_eval(raw_text)
        result = _extract_from_parsed(parsed)
        if result:
            return result
    except (ValueError, SyntaxError):
        pass

    # Strategy 3: Regex extraction from prose/unstructured text
    # Matches patterns like: SPEAKER_00: Ken Jennings, "SPEAKER_01" -> "Matt"
    pattern = re.compile(
        r'["\']?(SPEAKER_\d+)["\']?\s*[:\-=→>]+\s*["\']?([A-Z][a-zA-Z\s\'.]+?)(?:["\',}\]\n]|$)',
        re.MULTILINE,
    )
    matches = pattern.findall(raw_text)
    if matches:
        mapping: Dict[str, str] = {}
        for speaker_id, name in matches:
            name_clean = name.strip().rstrip(".,;:'\"")
            if name_clean:
                mapping[speaker_id] = name_clean
        if mapping:
            logger.info(
                "Pass 1: extracted speaker mapping via regex fallback",
                pairs_found=len(mapping),
            )
            return mapping

    logger.warning(
        "Pass 1: could not extract speaker mapping from response",
        raw_text=raw_text[:300],
    )
    return {}


def _extract_from_parsed(parsed: object) -> "Optional[Dict[str, str]]":
    """
    Extracts SPEAKER_XX → name pairs from a parsed JSON/Python structure.
    Handles dicts (flat or nested) and lists of dicts.
    """
    if isinstance(parsed, dict):
        # Check if it's a flat SPEAKER_XX → name dict
        if any(str(k).upper().startswith("SPEAKER_") for k in parsed):
            return {str(k): str(v).strip() for k, v in parsed.items() if isinstance(v, str) and v.strip()}
        # Check for nested structure like {"speakers": {...}, "mapping": {...}}
        for v in parsed.values():
            if isinstance(v, dict) and any(str(sk).upper().startswith("SPEAKER_") for sk in v):
                return {str(sk): str(sv).strip() for sk, sv in v.items() if isinstance(sv, str) and sv.strip()}

    if isinstance(parsed, list):
        # List of dicts: [{"speaker": "SPEAKER_00", "name": "Ken"}, ...]
        mapping: Dict[str, str] = {}
        is_segments = False
        for item in parsed:
            if not isinstance(item, dict):
                continue
            if "text" in item and "speaker" in item:
                is_segments = True
            # Try common key patterns
            speaker_key = None
            name_key = None
            for k in item:
                k_lower = str(k).lower()
                if k_lower in ("speaker", "speaker_id", "id"):
                    speaker_key = k
                elif k_lower in ("name", "contestant", "person"):
                    name_key = k
            if speaker_key and name_key:
                mapping[str(item[speaker_key])] = str(item[name_key]).strip()
        if mapping:
            return mapping
        elif is_segments:
            logger.warning(
                "Pass 1: LLM returned diarized segments instead of a mapping. Attempting heuristic inference."
            )
            inferred = _infer_speaker_mapping_from_segments(parsed)
            if inferred:
                return inferred

    return None


def _infer_speaker_mapping_from_segments(segments: list[Any]) -> Dict[str, str]:
    """
    Infer SPEAKER_XX → Name mapping from a list of diarized transcript segments.
    Fallback heuristic when the LLM returns the raw JSON segments instead of a mapping.
    """
    import re

    # Count segments per speaker
    speaker_counts: Dict[str, int] = {}
    speaker_texts: Dict[str, list[str]] = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        spk = seg.get("speaker", "")
        if not spk:
            continue
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        speaker_texts.setdefault(spk, []).append(str(seg.get("text", "")))

    if not speaker_counts:
        return {}

    # Host is the most frequent speaker (reads clues, narrates)
    host_speaker = max(speaker_counts, key=lambda k: speaker_counts[k])

    known_hosts = ["Ken Jennings", "Ryan Seacrest", "Mayim Bialik", "Alex Trebek", "Buzzy Cohen"]

    # Extract contestant names mentioned by the host
    host_text = " ".join(speaker_texts.get(host_speaker, []))
    name_pattern = re.findall(r"\b([A-Z][a-z]{2,})\b", host_text)

    common_words = {
        "The",
        "And",
        "But",
        "For",
        "Not",
        "You",
        "Your",
        "Yes",
        "That",
        "This",
        "What",
        "Who",
        "Where",
        "When",
        "How",
        "Which",
        "Here",
        "There",
        "Back",
        "Right",
        "Correct",
        "Answer",
        "Daily",
        "Double",
        "Jeopardy",
        "Final",
        "Category",
        "Categories",
        "Clue",
        "Score",
        "Place",
        "First",
        "Second",
        "Third",
        "Let",
        "Take",
        "Look",
        "Start",
        "Pick",
        "Select",
        "Got",
        "Put",
        "Going",
        "Come",
        "Most",
        "Last",
        "Next",
        "Okay",
        "Well",
        "Now",
        "Get",
        "See",
        "Say",
        "Try",
        "Make",
        "Give",
        "Just",
        "Also",
        "Still",
        "Much",
        "Many",
        "Some",
        "More",
        "Very",
        "Than",
        "Only",
        "Good",
        "Great",
        "Nice",
        "Excellent",
        "Exactly",
        "Time",
        "Round",
        "Board",
        "Please",
        "Today",
        "Thanks",
        "Thank",
        "These",
        "Those",
        "They",
        "Their",
        "Them",
        "Because",
        "Would",
        "Could",
        "Should",
        "Been",
        "Have",
        "Has",
        "Had",
        "Does",
        "Did",
    }
    name_freq: Dict[str, int] = {}
    for name in name_pattern:
        if name not in common_words and len(name) > 2:
            name_freq[name] = name_freq.get(name, 0) + 1

    # Top 3 most-mentioned names are likely contestants
    contestant_names = sorted(name_freq, key=lambda k: name_freq[k], reverse=True)[:3]

    mapping: Dict[str, str] = {}

    host_name = "Host"
    for known in known_hosts:
        if known.split()[0] in host_text or known.split()[-1] in host_text:
            host_name = known
            break
    mapping[host_speaker] = host_name

    # Match other speakers
    non_host_speakers = sorted(
        [s for s in speaker_counts if s != host_speaker],
        key=lambda s: int(s.replace("SPEAKER_", "")) if s.startswith("SPEAKER_") else 999,
    )

    used_names: set[str] = set()
    for spk in non_host_speakers:
        best_name = None
        for name in contestant_names:
            if name not in used_names:
                best_name = name
                break
        if best_name:
            mapping[spk] = best_name
            used_names.add(best_name)
        else:
            mapping[spk] = f"Contestant ({spk})"

    logger.info("Pass 1: inferred speaker mapping from segments", inferred_mapping=mapping)
    return mapping
