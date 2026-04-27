"""
Pipeline stage definitions — single source of truth for stage names,
activation maps, and upstream dependency chains.

Used by the orchestrator for worker dispatch and by the CLI for
``--stage`` flag validation.
"""

from typing import Set

# Valid stage names for the --stage CLI flag
VALID_STAGES = ("all", "transcribe", "extract", "augment", "verify")

# Which logical stages are active for each --stage value
ACTIVE_STAGES: dict[str, Set[str]] = {
    "all": {"transcribe", "extract", "augment", "verify"},
    "transcribe": {"transcribe"},
    "extract": {"extract"},
    "augment": {"augment"},
    "verify": {"verify"},
}

# Upstream status checking for --once mode termination.
# "full" = wait for upstream stages before exiting (used when all stages run together).
# "isolated" = only check own input/in-flight statuses (used for single-stage runs).
UPSTREAM_MAP_FULL: dict[str, list[str]] = {
    "PENDING": ["PENDING"],
    "TRANSCRIPT_READY": ["PENDING", "TRANSCRIBING", "TRANSCRIPT_READY", "CLEANED"],
    "SAVING": [
        "PENDING",
        "TRANSCRIBING",
        "TRANSCRIPT_READY",
        "CLEANED",
        "SAVING",
    ],
    "MULTIMODAL_DONE": [
        "PENDING",
        "TRANSCRIBING",
        "TRANSCRIPT_READY",
        "CLEANED",
        "SAVING",
        "MULTIMODAL_PROCESSING",
        "MULTIMODAL_DONE",
        "VECTORIZING",
    ],
}

UPSTREAM_MAP_ISOLATED: dict[str, list[str]] = {
    "PENDING": ["PENDING"],
    "TRANSCRIPT_READY": ["TRANSCRIPT_READY", "CLEANED"],
    "SAVING": ["SAVING", "MULTIMODAL_PROCESSING"],
    "MULTIMODAL_DONE": ["MULTIMODAL_DONE", "VECTORIZING"],
}
