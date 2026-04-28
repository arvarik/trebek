"""Tests for Pass 1 speaker mapping normalization — handles non-deterministic LLM output."""

from trebek.llm.pass1_anchoring import _normalize_speaker_mapping, _extract_from_parsed


class TestNormalizeSpeakerMapping:
    """Test robust parsing of LLM output into SPEAKER_XX → name dict."""

    def test_standard_json_dict(self) -> None:
        """Standard JSON dict should parse cleanly."""
        raw = '{"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Matt Amodio"}'
        result = _normalize_speaker_mapping(raw)
        assert result == {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Matt Amodio"}

    def test_single_quoted_dict(self) -> None:
        """Single-quoted Python dict (ast.literal_eval fallback)."""
        raw = "{'SPEAKER_00': 'Ken Jennings', 'SPEAKER_01': 'Matt Amodio'}"
        result = _normalize_speaker_mapping(raw)
        assert result == {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Matt Amodio"}

    def test_list_of_dicts(self) -> None:
        """List-of-dicts format should be normalized to flat dict."""
        raw = '[{"speaker": "SPEAKER_00", "name": "Ken"}, {"speaker_id": "SPEAKER_01", "name": "Matt"}]'
        result = _normalize_speaker_mapping(raw)
        assert result["SPEAKER_00"] == "Ken"

    def test_nested_dict(self) -> None:
        """Nested structure like {speakers: {...}} should be unwrapped."""
        raw = '{"speakers": {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Matt"}}'
        result = _normalize_speaker_mapping(raw)
        assert result == {"SPEAKER_00": "Ken Jennings", "SPEAKER_01": "Matt"}

    def test_prose_with_patterns(self) -> None:
        """Regex fallback should extract from prose text."""
        raw = "SPEAKER_00: Ken Jennings\nSPEAKER_01: Matt Amodio"
        result = _normalize_speaker_mapping(raw)
        assert "SPEAKER_00" in result
        assert "Ken Jennings" in result["SPEAKER_00"]

    def test_empty_string(self) -> None:
        """Empty/garbage input should return empty dict, not crash."""
        assert _normalize_speaker_mapping("") == {}
        assert _normalize_speaker_mapping("no speakers here") == {}

    def test_filters_empty_values(self) -> None:
        """Empty string values should be filtered out."""
        raw = '{"SPEAKER_00": "Ken", "SPEAKER_01": "", "SPEAKER_02": "Matt"}'
        result = _normalize_speaker_mapping(raw)
        assert "SPEAKER_01" not in result
        assert result["SPEAKER_00"] == "Ken"
        assert result["SPEAKER_02"] == "Matt"


class TestExtractFromParsed:
    """Test the structural extraction helper."""

    def test_flat_dict(self) -> None:
        parsed = {"SPEAKER_00": "Ken", "SPEAKER_01": "Matt"}
        assert _extract_from_parsed(parsed) == {"SPEAKER_00": "Ken", "SPEAKER_01": "Matt"}

    def test_non_speaker_dict(self) -> None:
        """Dict without SPEAKER_ keys should return None."""
        assert _extract_from_parsed({"name": "Ken"}) is None

    def test_list_format(self) -> None:
        parsed = [{"speaker": "SPEAKER_00", "name": "Ken"}, {"speaker": "SPEAKER_01", "name": "Matt"}]
        result = _extract_from_parsed(parsed)
        assert result is not None
        assert result["SPEAKER_00"] == "Ken"

    def test_empty_list(self) -> None:
        assert _extract_from_parsed([]) is None

    def test_non_dict_non_list(self) -> None:
        assert _extract_from_parsed("just a string") is None
        assert _extract_from_parsed(42) is None
