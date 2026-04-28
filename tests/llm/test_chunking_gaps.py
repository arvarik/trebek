"""
Tests for chunking coverage gaps — targets the fallback path (lines 41-50)
where a large transcript with no round markers triggers sized chunking,
and verifies overlap behavior in the fallback.
"""

from trebek.llm.chunking import _chunk_by_semantic_boundaries


class TestFallbackSizedChunking:
    """The fallback path at lines 41-50: large transcript, no markers, single raw chunk."""

    def test_fallback_triggers_on_large_single_chunk(self) -> None:
        """A large transcript with exactly one raw chunk (no markers) triggers fallback."""
        # Create a transcript larger than max_chunk_lines but with no round markers
        lines = [f"L{i} S0: generic line {i}" for i in range(3000)]
        chunks = _chunk_by_semantic_boundaries(lines, max_chunk_lines=200)
        # Should have multiple chunks (3000/200 = ~15+ with overlap)
        assert len(chunks) > 10

    def test_fallback_chunks_have_overlap(self) -> None:
        """Fallback chunks should maintain 40-line overlap."""
        lines = [f"L{i} S0: generic line {i}" for i in range(500)]
        chunks = _chunk_by_semantic_boundaries(lines, max_chunk_lines=200)
        # Verify overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk_a_lines = chunks[i].split("\n")
            chunk_b_lines = chunks[i + 1].split("\n")
            # Last 40 lines of chunk A should appear in chunk B
            overlap = set(chunk_a_lines[-40:]) & set(chunk_b_lines[:80])
            assert len(overlap) > 0, f"No overlap between chunk {i} and {i + 1}"

    def test_fallback_covers_all_content(self) -> None:
        """Fallback chunking should not lose any lines."""
        lines = [f"L{i} S0: line {i}" for i in range(500)]
        chunks = _chunk_by_semantic_boundaries(lines, max_chunk_lines=200)
        # Every original line should appear in at least one chunk
        all_chunk_content = "\n".join(chunks)
        for line in lines:
            assert line in all_chunk_content


class TestLargeRoundSubChunkOverlap:
    """Tests for the overlap within a single large round (lines 32-39)."""

    def test_single_large_round_subchunked_with_overlap(self) -> None:
        """A single round exceeding max_chunk_lines should be split with 40-line overlap."""
        # Make a transcript with markers early, then a huge second round
        lines = [f"L{i} S0: Jeopardy text {i}" for i in range(50)]
        lines.append("L50 S0: And now, Double Jeopardy!")
        lines.extend([f"L{i} S0: DJ text {i}" for i in range(51, 551)])  # 500 lines

        chunks = _chunk_by_semantic_boundaries(lines, max_chunk_lines=200)
        # First chunk = Jeopardy round (51 lines), then DJ sub-chunks
        assert len(chunks) >= 4  # 51 + 500/200 = at least 4

    def test_tiebreaker_marker_splits(self) -> None:
        """Tiebreaker round marker should trigger a chunk boundary."""
        lines = [f"L{i} S0: text {i}" for i in range(50)]
        lines.append("L50 S0: We have a tiebreaker round!")
        lines.extend([f"L{i} S0: tiebreaker text {i}" for i in range(51, 70)])
        chunks = _chunk_by_semantic_boundaries(lines)
        assert len(chunks) >= 2

    def test_exactly_max_chunk_lines_no_split(self) -> None:
        """A round with exactly max_chunk_lines should NOT be sub-chunked."""
        lines = [f"L{i} S0: text {i}" for i in range(200)]
        chunks = _chunk_by_semantic_boundaries(lines, max_chunk_lines=200)
        assert len(chunks) == 1

    def test_max_chunk_lines_plus_one_triggers_split(self) -> None:
        """A round with max_chunk_lines + 1 should be sub-chunked."""
        lines = [f"L{i} S0: text {i}" for i in range(201)]
        chunks = _chunk_by_semantic_boundaries(lines, max_chunk_lines=200)
        assert len(chunks) == 2


class TestSingleLineTranscript:
    """Edge case: transcript with exactly one line."""

    def test_single_line(self) -> None:
        chunks = _chunk_by_semantic_boundaries(["single line"])
        assert len(chunks) == 1
        assert chunks[0] == "single line"
