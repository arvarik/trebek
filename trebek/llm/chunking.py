import structlog

logger = structlog.get_logger()

_ROUND_MARKERS = [
    "double jeopardy",
    "final jeopardy",
    "tiebreaker",
]


def _chunk_by_semantic_boundaries(transcript_lines: list[str], max_chunk_lines: int = 2000) -> list[str]:
    raw_chunks: list[list[str]] = []
    current_chunk: list[str] = []

    for line in transcript_lines:
        current_chunk.append(line)
        lower_line = line.lower()
        if any(marker in lower_line for marker in _ROUND_MARKERS) and len(current_chunk) > 10:
            raw_chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        raw_chunks.append(current_chunk)

    final_chunks: list[str] = []
    overlap = 40

    for chunk_lines in raw_chunks:
        if len(chunk_lines) <= max_chunk_lines:
            final_chunks.append("\n".join(chunk_lines))
        else:
            i = 0
            while i < len(chunk_lines):
                end = min(i + max_chunk_lines, len(chunk_lines))
                final_chunks.append("\n".join(chunk_lines[i:end]))
                i += max_chunk_lines - overlap
                if i >= len(chunk_lines):
                    break

    if len(final_chunks) == 1 and len(transcript_lines) > max_chunk_lines:
        logger.warning("No round markers found, falling back to sized chunking")
        final_chunks = []
        i = 0
        while i < len(transcript_lines):
            end = min(i + max_chunk_lines, len(transcript_lines))
            final_chunks.append("\n".join(transcript_lines[i:end]))
            i += max_chunk_lines - overlap
            if i >= len(transcript_lines):
                break

    return final_chunks
