"""Tests for _try_close_truncated_json — the critical safety net for MAX_TOKENS truncation."""

from trebek.llm.utils import _try_close_truncated_json
import json


class TestTryCloseTruncatedJson:
    def test_already_complete_json_returned_unchanged(self) -> None:
        """Complete JSON should be returned as-is."""
        complete = '{"clues": [{"text": "hello"}]}'
        result = _try_close_truncated_json(complete)
        assert result == complete
        json.loads(result)  # Must be valid JSON

    def test_truncated_mid_array(self) -> None:
        """JSON cut mid-array should close brackets and braces."""
        truncated = '{"clues": [{"text": "hello"}, {"text": "world"'
        result = _try_close_truncated_json(truncated)
        parsed = json.loads(result)
        assert "clues" in parsed

    def test_truncated_inside_string(self) -> None:
        """JSON cut inside a string value should close the string first."""
        truncated = '{"clues": [{"text": "this is a long clue about hist'
        result = _try_close_truncated_json(truncated)
        parsed = json.loads(result)
        assert "clues" in parsed

    def test_brackets_inside_strings_not_counted(self) -> None:
        """Brackets/braces inside string values must NOT affect structural counting."""
        truncated = '{"text": "array [1, 2, 3] and object {a: b}", "other": "val'
        result = _try_close_truncated_json(truncated)
        parsed = json.loads(result)
        assert parsed["text"] == "array [1, 2, 3] and object {a: b}"

    def test_trailing_partial_key_stripped(self) -> None:
        """A trailing partial key-value (no value yet) should be stripped."""
        truncated = '{"clues": [{"text": "hello", "category": '
        result = _try_close_truncated_json(truncated)
        parsed = json.loads(result)
        assert "clues" in parsed

    def test_trailing_comma_stripped(self) -> None:
        """A trailing comma should be stripped before closing."""
        truncated = '{"clues": [{"text": "hello"},'
        result = _try_close_truncated_json(truncated)
        parsed = json.loads(result)
        assert len(parsed["clues"]) == 1

    def test_empty_string_does_not_crash(self) -> None:
        """Empty input should not crash."""
        result = _try_close_truncated_json("")
        # Empty string won't produce valid JSON but shouldn't crash
        assert isinstance(result, str)

    def test_escaped_quotes_inside_string(self) -> None:
        """Escaped quotes inside strings should not toggle in_string state."""
        truncated = '{"text": "he said \\"hello\\"", "other": "v'
        result = _try_close_truncated_json(truncated)
        parsed = json.loads(result)
        assert 'he said "hello"' in parsed["text"]

    def test_deeply_nested_truncation(self) -> None:
        """Deeply nested JSON truncated mid-value should close all levels."""
        truncated = '{"a": {"b": {"c": [{"d": "val'
        result = _try_close_truncated_json(truncated)
        parsed = json.loads(result)
        assert "a" in parsed

    def test_realistic_clue_truncation(self) -> None:
        """Simulate a realistic PartialClues truncation mid-clue."""
        truncated = (
            '{"clues": ['
            '{"round": "J!", "category": "History", "board_row": 1, "board_col": 1, '
            '"is_daily_double": false, "requires_visual_context": false, '
            '"host_read_start_line_id": "L5", "host_read_end_line_id": "L8", '
            '"correct_response": "What is Rome?", "attempts": []}, '
            '{"round": "J!", "category": "Science", "board_row": 2, "board_col": 1, '
            '"is_daily_double": false, "requires_visual_context": false, '
            '"host_read_start_line_id": "L15", "host_read_end_line_id": "L18", '
            '"correct_response": "What is'
        )
        result = _try_close_truncated_json(truncated)
        parsed = json.loads(result)
        # Should recover at least the first complete clue
        assert len(parsed["clues"]) >= 1
        assert parsed["clues"][0]["category"] == "History"
