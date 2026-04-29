"""Tests for semantic chunking logic — the core of transcript splitting for parallel extraction."""

from trebek.llm.chunking import _chunk_by_semantic_boundaries, split_transcript_by_round


def test_single_round_no_split() -> None:
    """A short transcript with no round markers should return a single chunk."""
    lines = [f"L{i} S0: Some text line {i}" for i in range(50)]
    chunks = _chunk_by_semantic_boundaries(lines)
    assert len(chunks) == 1
    assert "L0" in chunks[0]
    assert "L49" in chunks[0]


def test_splits_on_double_jep_marker() -> None:
    """Transcript containing 'double j!' should split into 2+ chunks."""
    lines = [f"L{i} S0: J! clue text {i}" for i in range(50)]
    lines.append("L50 S0: And now, Double J!")
    lines.extend([f"L{i} S0: DJ clue text {i}" for i in range(51, 100)])
    chunks = _chunk_by_semantic_boundaries(lines)
    assert len(chunks) >= 2
    # First chunk should contain J! content
    assert "J! clue text" in chunks[0]
    # Second chunk should start with or include the DJ marker
    assert "Double J!" in chunks[0] or "Double J!" in chunks[1]


def test_splits_on_final_jep_marker() -> None:
    """Transcript with Final J! marker should produce a chunk boundary."""
    lines = [f"L{i} S0: Some text {i}" for i in range(30)]
    lines.append("L30 S0: It's time for Final J!")
    lines.extend([f"L{i} S0: FJ text {i}" for i in range(31, 50)])
    chunks = _chunk_by_semantic_boundaries(lines)
    assert len(chunks) >= 2


def test_large_chunk_gets_subchunked() -> None:
    """A single round exceeding max_chunk_lines should be split into sub-chunks."""
    lines = [f"L{i} S0: Very long round text line {i}" for i in range(500)]
    chunks = _chunk_by_semantic_boundaries(lines, max_chunk_lines=200)
    assert len(chunks) >= 3  # 500 lines / 200 max = at least 3 chunks


def test_overlap_between_subchunks() -> None:
    """Sub-chunks should have 40-line overlap to prevent boundary loss."""
    lines = [f"L{i} S0: line {i}" for i in range(500)]
    chunks = _chunk_by_semantic_boundaries(lines, max_chunk_lines=200)
    # Check that chunks overlap: last lines of chunk[0] appear in chunk[1]
    chunk0_lines = chunks[0].split("\n")
    chunk1_lines = chunks[1].split("\n")
    # The last 40 lines of chunk 0 should appear as the first 40 of chunk 1
    assert chunk0_lines[-1] in chunk1_lines


def test_no_round_markers_falls_back_to_sized_chunking() -> None:
    """Without round markers on a large transcript, falls back to sized chunking."""
    lines = [f"L{i} S0: plain line {i}" for i in range(500)]
    chunks = _chunk_by_semantic_boundaries(lines, max_chunk_lines=200)
    assert len(chunks) >= 3


def test_round_marker_ignored_if_chunk_too_small() -> None:
    """A round marker within the first 10 lines should NOT trigger a split (anti-false-positive)."""
    lines = ["L0 S0: double j! in intro"]
    lines.extend([f"L{i} S0: rest of text {i}" for i in range(1, 50)])
    chunks = _chunk_by_semantic_boundaries(lines)
    # The marker at line 0 should be ignored because chunk < 10 lines
    assert len(chunks) == 1


def test_empty_transcript() -> None:
    """An empty transcript should return an empty list."""
    chunks = _chunk_by_semantic_boundaries([])
    assert len(chunks) == 0


# ── split_transcript_by_round tests ──────────────────────────────────


def _make_round_transcript(j_lines: int = 200, dj_lines: int = 200, fj_lines: int = 50) -> list[str]:
    """Generate a mock transcript with round markers."""
    lines = []
    for i in range(j_lines):
        lines.append(f"L{i} S0: J! round clue text line {i}")
    lines.append(f"L{j_lines} S0: And now we move to Double J!")
    for i in range(dj_lines):
        idx = j_lines + 1 + i
        lines.append(f"L{idx} S0: DJ! round clue text line {i}")
    fj_start = j_lines + 1 + dj_lines
    lines.append(f"L{fj_start} S0: It's time for Final J!")
    for i in range(fj_lines):
        idx = fj_start + 1 + i
        lines.append(f"L{idx} S0: FJ content line {i}")
    return lines


def test_split_into_three_regions() -> None:
    lines = _make_round_transcript()
    j_text, dj_text, fj_text = split_transcript_by_round(lines)
    assert j_text
    assert dj_text
    assert fj_text


def test_split_j_text_contains_j_clues() -> None:
    lines = _make_round_transcript()
    j_text, _, _ = split_transcript_by_round(lines)
    assert "J! round clue text" in j_text


def test_split_dj_text_contains_dj_clues() -> None:
    lines = _make_round_transcript()
    _, dj_text, _ = split_transcript_by_round(lines)
    assert "DJ! round clue text" in dj_text


def test_split_fj_text_contains_fj_content() -> None:
    lines = _make_round_transcript()
    _, _, fj_text = split_transcript_by_round(lines)
    assert "Final J!" in fj_text


def test_split_no_dj_marker_returns_full_transcript() -> None:
    lines = [f"L{i} S0: Some text line {i}" for i in range(100)]
    j_text, dj_text, fj_text = split_transcript_by_round(lines)
    assert j_text
    assert dj_text == ""
    assert fj_text == ""


def test_split_overlap_extends_boundaries() -> None:
    lines = _make_round_transcript(j_lines=100, dj_lines=100)
    j_text, dj_text, _ = split_transcript_by_round(lines, overlap_lines=20)
    j_lines_count = len(j_text.split("\n"))
    assert j_lines_count > 100  # Must exceed J! line count due to overlap


def test_split_zero_overlap() -> None:
    lines = _make_round_transcript(j_lines=100, dj_lines=100)
    j_text, dj_text, _ = split_transcript_by_round(lines, overlap_lines=0)
    j_set = set(j_text.split("\n"))
    dj_set = set(dj_text.split("\n"))
    overlap = j_set & dj_set
    assert len(overlap) <= 1


def test_split_early_dj_marker_ignored() -> None:
    """DJ! marker within first 10 lines should be ignored (false positive)."""
    lines = ["L0 S0: Welcome to Double J! preview"] + [f"L{i} S0: intro line {i}" for i in range(1, 5)]
    lines += [f"L{i} S0: J! content {i}" for i in range(5, 200)]
    lines.append("L200 S0: Now it's time for Double J!")
    lines += [f"L{i} S0: DJ content {i}" for i in range(201, 300)]
    j_text, dj_text, _ = split_transcript_by_round(lines)
    assert "DJ content" in dj_text
    assert "J! content" in j_text
