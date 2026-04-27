"""Tests for semantic chunking logic — the core of transcript splitting for parallel extraction."""

from trebek.llm.chunking import _chunk_by_semantic_boundaries


def test_single_round_no_split() -> None:
    """A short transcript with no round markers should return a single chunk."""
    lines = [f"L{i} S0: Some text line {i}" for i in range(50)]
    chunks = _chunk_by_semantic_boundaries(lines)
    assert len(chunks) == 1
    assert "L0" in chunks[0]
    assert "L49" in chunks[0]


def test_splits_on_double_jeopardy_marker() -> None:
    """Transcript containing 'double jeopardy' should split into 2+ chunks."""
    lines = [f"L{i} S0: Jeopardy clue text {i}" for i in range(50)]
    lines.append("L50 S0: And now, Double Jeopardy!")
    lines.extend([f"L{i} S0: DJ clue text {i}" for i in range(51, 100)])
    chunks = _chunk_by_semantic_boundaries(lines)
    assert len(chunks) >= 2
    # First chunk should contain Jeopardy content
    assert "Jeopardy clue text" in chunks[0]
    # Second chunk should start with or include the DJ marker
    assert "Double Jeopardy" in chunks[0] or "Double Jeopardy" in chunks[1]


def test_splits_on_final_jeopardy_marker() -> None:
    """Transcript with Final Jeopardy marker should produce a chunk boundary."""
    lines = [f"L{i} S0: Some text {i}" for i in range(30)]
    lines.append("L30 S0: It's time for Final Jeopardy!")
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
    lines = ["L0 S0: double jeopardy in intro"]
    lines.extend([f"L{i} S0: rest of text {i}" for i in range(1, 50)])
    chunks = _chunk_by_semantic_boundaries(lines)
    # The marker at line 0 should be ignored because chunk < 10 lines
    assert len(chunks) == 1


def test_empty_transcript() -> None:
    """An empty transcript should return an empty list."""
    chunks = _chunk_by_semantic_boundaries([])
    assert len(chunks) == 0
