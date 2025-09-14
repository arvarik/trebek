from __future__ import annotations

import json
import os
from datetime import datetime, timezone
import re
from typing import Any, Dict

import config


def _safe_filename(stem: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in stem)


def save_transcription(
    source_video_path: str, transcription: Dict[str, Any]
) -> str:
    """Persist transcription to a JSON file under TRANSCRIPTS_DIR.

    Returns the absolute path to the saved JSON file.
    """
    os.makedirs(config.TRANSCRIPTS_DIR, exist_ok=True)

    basename = os.path.basename(source_video_path)
    name_wo_ext = os.path.splitext(basename)[0]

    # Extract season/episode from filename (assumes presence of S#E#)
    season: int | None = None
    episode: int | None = None
    m = re.search(r"[sS]\s*(\d+)\s*[eE]\s*(\d+)", name_wo_ext)
    if m:
        season = int(m.group(1))
        episode = int(m.group(2))

    # Build output filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if season is not None and episode is not None:
        base_filename = _safe_filename(f"Jeopardy S{season}E{episode}.{timestamp}")
    else:
        base_filename = _safe_filename(f"{name_wo_ext}.{timestamp}")
    filename = f"{base_filename}.json"
    output_path = os.path.join(config.TRANSCRIPTS_DIR, filename)

    if os.path.exists(output_path):
        counter = 1
        while True:
            alt_filename = f"{base_filename}.{counter}.json"
            alt_path = os.path.join(config.TRANSCRIPTS_DIR, alt_filename)
            if not os.path.exists(alt_path):
                output_path = alt_path
                break
            counter += 1

    payload = {
        "source_video": os.path.abspath(source_video_path),
        "created_utc": timestamp,
        **transcription,
    }

    # Include season/episode if found (as integers)
    if season is not None and episode is not None:
        payload["season"] = season
        payload["episode"] = episode

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path


