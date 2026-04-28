"""
Pipeline stage definitions — single source of truth for stage names,
activation maps, and upstream dependency chains.

Used by the orchestrator for worker dispatch and by the CLI for
``--stage`` flag validation.
"""

from typing import Set

from trebek.status import PipelineStatus as S

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
    S.PENDING: [S.PENDING],
    S.TRANSCRIPT_READY: [S.PENDING, S.TRANSCRIBING, S.TRANSCRIPT_READY, S.CLEANED],
    S.SAVING: [
        S.PENDING,
        S.TRANSCRIBING,
        S.TRANSCRIPT_READY,
        S.CLEANED,
        S.SAVING,
    ],
    S.MULTIMODAL_DONE: [
        S.PENDING,
        S.TRANSCRIBING,
        S.TRANSCRIPT_READY,
        S.CLEANED,
        S.SAVING,
        S.MULTIMODAL_PROCESSING,
        S.MULTIMODAL_DONE,
        S.VECTORIZING,
    ],
}

UPSTREAM_MAP_ISOLATED: dict[str, list[str]] = {
    S.PENDING: [S.PENDING],
    S.TRANSCRIPT_READY: [S.TRANSCRIPT_READY, S.CLEANED],
    S.SAVING: [S.SAVING, S.MULTIMODAL_PROCESSING],
    S.MULTIMODAL_DONE: [S.MULTIMODAL_DONE, S.VECTORIZING],
}
