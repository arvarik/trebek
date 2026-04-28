"""
Tests for pass1 speaker mapping normalization — pure parsing functions
that handle multiple LLM output formats without needing API calls.
"""

from trebek.llm.pass1_anchoring import (
    _normalize_speaker_mapping,
    _extract_from_parsed,
)


class TestNormalizeSpeakerMapping:
    """Tests for _normalize_speaker_mapping multi-format parser."""

    def test_standard_json_dict(self) -> None:
        raw = '{"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Matt Amodio"}'
        result = _normalize_speaker_mapping(raw)
        assert result == {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Matt Amodio"}

    def test_single_quoted_python_dict(self) -> None:
        raw = "{'SPEAKER_00': 'Ken Jennings', 'SPEAKER_01': 'Matt'}"
        result = _normalize_speaker_mapping(raw)
        assert result == {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Matt"}

    def test_prose_regex_fallback(self) -> None:
        raw = """Based on the audio analysis:
SPEAKER_00: Ken Jennings
SPEAKER_01: Rachel Bernstein
SPEAKER_02: Dan Puma"""
        result = _normalize_speaker_mapping(raw)
        assert "SPEAKER_00" in result
        assert "SPEAKER_01" in result
        assert result["SPEAKER_00"] == "Ken Jennings"
        assert result["SPEAKER_01"] == "Rachel Bernstein"

    def test_unparseable_returns_empty(self) -> None:
        raw = "I couldn't determine the speakers from this audio."
        result = _normalize_speaker_mapping(raw)
        assert result == {}

    def test_markdown_fenced_json(self) -> None:
        """Already cleaned by caller, but test the inner JSON parsing."""
        raw = '{"SPEAKER_00": "Host", "SPEAKER_01": "Alice"}'
        result = _normalize_speaker_mapping(raw)
        assert result["SPEAKER_01"] == "Alice"

    def test_nested_dict_structure(self) -> None:
        raw = '{"speakers": {"SPEAKER_00": "Ken", "SPEAKER_01": "Alice"}}'
        result = _normalize_speaker_mapping(raw)
        assert result == {"SPEAKER_00": "Ken", "SPEAKER_01": "Alice"}

    def test_arrow_syntax_in_prose(self) -> None:
        raw = "SPEAKER_00 -> Ken Jennings\nSPEAKER_01 -> Rachel"
        result = _normalize_speaker_mapping(raw)
        assert "SPEAKER_00" in result
        assert "SPEAKER_01" in result

    def test_equals_syntax_in_prose(self) -> None:
        raw = "SPEAKER_00 = Ken Jennings\nSPEAKER_01 = Rachel"
        result = _normalize_speaker_mapping(raw)
        assert "SPEAKER_00" in result

    def test_whitespace_stripped_from_names(self) -> None:
        raw = '{"SPEAKER_00": "  Ken Jennings  ", "SPEAKER_01": "  Alice  "}'
        result = _normalize_speaker_mapping(raw)
        assert result["SPEAKER_00"] == "Ken Jennings"
        assert result["SPEAKER_01"] == "Alice"

    def test_empty_values_excluded(self) -> None:
        raw = '{"SPEAKER_00": "Ken", "SPEAKER_01": ""}'
        result = _normalize_speaker_mapping(raw)
        assert "SPEAKER_01" not in result


class TestExtractFromParsed:
    """Tests for _extract_from_parsed — handles dict and list formats."""

    def test_flat_speaker_dict(self) -> None:
        parsed = {"SPEAKER_00": "Ken", "SPEAKER_01": "Alice"}
        result = _extract_from_parsed(parsed)
        assert result == {"SPEAKER_00": "Ken", "SPEAKER_01": "Alice"}

    def test_nested_speaker_dict(self) -> None:
        parsed = {"mapping": {"SPEAKER_00": "Ken", "SPEAKER_01": "Alice"}}
        result = _extract_from_parsed(parsed)
        assert result == {"SPEAKER_00": "Ken", "SPEAKER_01": "Alice"}

    def test_list_of_dicts_with_speaker_name_keys(self) -> None:
        parsed = [
            {"speaker": "SPEAKER_00", "name": "Ken"},
            {"speaker": "SPEAKER_01", "name": "Alice"},
        ]
        result = _extract_from_parsed(parsed)
        assert result == {"SPEAKER_00": "Ken", "SPEAKER_01": "Alice"}

    def test_list_of_dicts_with_id_contestant_keys(self) -> None:
        parsed = [
            {"id": "SPEAKER_00", "contestant": "Ken"},
            {"id": "SPEAKER_01", "contestant": "Alice"},
        ]
        result = _extract_from_parsed(parsed)
        assert result == {"SPEAKER_00": "Ken", "SPEAKER_01": "Alice"}

    def test_non_speaker_dict_returns_none(self) -> None:
        parsed = {"host": "Ken", "show": "J!"}
        result = _extract_from_parsed(parsed)
        assert result is None

    def test_empty_dict_returns_none(self) -> None:
        result = _extract_from_parsed({})
        assert result is None

    def test_empty_list_returns_none(self) -> None:
        result = _extract_from_parsed([])
        assert result is None

    def test_non_dict_non_list_returns_none(self) -> None:
        result = _extract_from_parsed("just a string")
        assert result is None

    def test_diarized_segments_fallback(self) -> None:
        """When LLM returns segments with 'text' + 'speaker' keys, trigger segment inference."""
        parsed = [
            {"text": "Welcome to the show", "speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
            {"text": "I'm from New York", "speaker": "SPEAKER_01", "start": 2.0, "end": 4.0},
        ]
        # This triggers the segment inference path but may not extract names
        result = _extract_from_parsed(parsed)
        # Even if no names are extracted, the function should return None (not crash)
        assert result is None or isinstance(result, dict)
