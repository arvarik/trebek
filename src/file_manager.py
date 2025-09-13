from __future__ import annotations

import json
import os
from datetime import datetime
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

    source_name = os.path.splitext(os.path.basename(source_video_path))[0]
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = _safe_filename(f"{source_name}.{timestamp}.json")
    output_path = os.path.join(config.TRANSCRIPTS_DIR, filename)

    payload = {
        "source_video": os.path.abspath(source_video_path),
        "created_utc": timestamp,
        **transcription,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path


