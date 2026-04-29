import structlog

logger = structlog.get_logger()

_ROUND_MARKERS = [
    "double j!",
    "final j!",
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


def split_transcript_by_round(
    transcript_lines: list[str],
    overlap_lines: int = 20,
) -> tuple[str, str, str]:
    """Split a transcript into J!, DJ!, and FJ regions at round boundaries.

    Finds the first occurrence of "Double J!" and "Final J!" markers to
    create clean round boundaries. An overlap region ensures clues near
    the boundary aren't missed by either extraction call.

    Returns:
        Tuple of (j_round_text, dj_round_text, fj_round_text).
        If a round boundary isn't found, returns the full transcript for
        the first region and empty strings for the rest.
    """
    dj_boundary: int | None = None
    fj_boundary: int | None = None

    for i, line in enumerate(transcript_lines):
        lower = line.lower()
        if dj_boundary is None and "double j!" in lower and i > 10:
            dj_boundary = i
        elif dj_boundary is not None and fj_boundary is None and "final j!" in lower and i > dj_boundary + 10:
            fj_boundary = i

    if dj_boundary is None:
        logger.warning(
            "No DJ! boundary found — cannot split by round, returning full transcript",
            total_lines=len(transcript_lines),
        )
        return "\n".join(transcript_lines), "", ""

    # J! region: start → DJ! boundary (+ overlap into DJ!)
    j_end = min(dj_boundary + overlap_lines, len(transcript_lines))
    j_text = "\n".join(transcript_lines[:j_end])

    # DJ! region: DJ! boundary (- overlap from J!) → FJ boundary (+ overlap)
    dj_start = max(0, dj_boundary - overlap_lines)
    if fj_boundary is not None:
        dj_end = min(fj_boundary + overlap_lines, len(transcript_lines))
        fj_text = "\n".join(transcript_lines[fj_boundary:])
    else:
        dj_end = len(transcript_lines)
        fj_text = ""
    dj_text = "\n".join(transcript_lines[dj_start:dj_end])

    logger.info(
        "Transcript split by round",
        j_lines=j_end,
        dj_lines=dj_end - dj_start,
        fj_lines=len(transcript_lines) - (fj_boundary or len(transcript_lines)),
        dj_boundary=dj_boundary,
        fj_boundary=fj_boundary,
        overlap=overlap_lines,
    )

    return j_text, dj_text, fj_text
