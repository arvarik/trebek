"""Tests for transcript formatting and speaker abbreviation utilities."""

from trebek.llm.transcript import (
    _abbreviate_speaker,
    _build_speaker_abbreviation_map,
    _format_transcript_compressed,
)


class TestAbbreviateSpeaker:
    def test_standard_speaker_id(self) -> None:
        assert _abbreviate_speaker("SPEAKER_00") == "S00"

    def test_double_digit_speaker_id(self) -> None:
        assert _abbreviate_speaker("SPEAKER_10") == "S10"

    def test_non_speaker_prefix_unchanged(self) -> None:
        assert _abbreviate_speaker("HOST") == "HOST"

    def test_empty_string(self) -> None:
        assert _abbreviate_speaker("") == ""

    def test_partial_prefix(self) -> None:
        """SPEAK (not SPEAKER_) should not be abbreviated."""
        assert _abbreviate_speaker("SPEAK_00") == "SPEAK_00"


class TestBuildSpeakerAbbreviationMap:
    def test_builds_map_from_segments(self) -> None:
        segments = [
            {"speaker": "SPEAKER_00", "text": "Hello"},
            {"speaker": "SPEAKER_01", "text": "World"},
            {"speaker": "SPEAKER_00", "text": "Again"},
        ]
        result = _build_speaker_abbreviation_map(segments)
        assert result == {"S00": "SPEAKER_00", "S01": "SPEAKER_01"}

    def test_unknown_speaker_not_in_map(self) -> None:
        """Non-SPEAKER_ speakers should not appear in abbreviation map."""
        segments = [{"speaker": "UNKNOWN", "text": "test"}]
        result = _build_speaker_abbreviation_map(segments)
        assert result == {}

    def test_missing_speaker_key(self) -> None:
        segments = [{"text": "no speaker here"}]
        result = _build_speaker_abbreviation_map(segments)
        # "UNKNOWN" is not SPEAKER_ prefixed, so empty map
        assert result == {}


class TestFormatTranscriptCompressed:
    def test_basic_formatting(self) -> None:
        segments = [
            {"speaker": "SPEAKER_00", "text": "Welcome to J!", "start": 0.0, "end": 1.5},
            {"speaker": "SPEAKER_01", "text": "Thank you", "start": 1.5, "end": 2.0},
        ]
        result = _format_transcript_compressed(segments)
        assert result == "L0 S00: Welcome to J!\nL1 S01: Thank you"

    def test_line_id_indexing(self) -> None:
        """Line IDs should be 0-indexed and sequential."""
        segments = [{"speaker": "SPEAKER_00", "text": f"Line {i}"} for i in range(5)]
        result = _format_transcript_compressed(segments)
        lines = result.split("\n")
        for i, line in enumerate(lines):
            assert line.startswith(f"L{i} ")

    def test_missing_text_key(self) -> None:
        """Segments with missing 'text' should produce empty text after speaker."""
        segments = [{"speaker": "SPEAKER_00"}]
        result = _format_transcript_compressed(segments)
        # .get("text", "") returns "", .strip() → "", so format is "L0 S00: "
        assert result == "L0 S00: "

    def test_missing_speaker_key(self) -> None:
        """Segments with missing 'speaker' should use UNKNOWN."""
        segments = [{"text": "some text"}]
        result = _format_transcript_compressed(segments)
        assert result == "L0 UNKNOWN: some text"

    def test_empty_segments(self) -> None:
        result = _format_transcript_compressed([])
        assert result == ""

    def test_text_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace in text should be stripped."""
        segments = [{"speaker": "SPEAKER_00", "text": "  padded text  "}]
        result = _format_transcript_compressed(segments)
        assert result == "L0 S00: padded text"

    def test_timestamps_not_in_output(self) -> None:
        """Timestamps should be omitted from compressed format."""
        segments = [{"speaker": "SPEAKER_00", "text": "Hello", "start": 5.123, "end": 6.456}]
        result = _format_transcript_compressed(segments)
        assert "5.123" not in result
        assert "6.456" not in result
