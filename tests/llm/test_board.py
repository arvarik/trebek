"""Unit tests for trebek.llm.board module.

Tests board format detection, value-to-row inference, dollar value parsing,
and round manifest construction.
"""

import pytest
from trebek.llm.board import (
    STANDARD_BOARD,
    CELEBRITY_BOARD,
    detect_board_format,
    infer_board_row,
    _parse_dollar_value,
    infer_board_row_from_selection_text,
    RoundManifest,
    build_manifests,
)


class TestBoardFormats:
    """Verify board format constants."""

    def test_standard_board_j_values(self) -> None:
        assert STANDARD_BOARD.j_values == (200, 400, 600, 800, 1000)

    def test_standard_board_dj_values(self) -> None:
        assert STANDARD_BOARD.dj_values == (400, 800, 1200, 1600, 2000)

    def test_celebrity_board_j_values(self) -> None:
        assert CELEBRITY_BOARD.j_values == (100, 200, 300, 400, 500)

    def test_celebrity_board_has_triple_j(self) -> None:
        assert CELEBRITY_BOARD.tj_values == (300, 600, 900, 1200, 1500)

    def test_standard_board_no_triple_j(self) -> None:
        assert STANDARD_BOARD.tj_values == ()

    def test_board_format_frozen(self) -> None:
        with pytest.raises(AttributeError):
            STANDARD_BOARD.name = "modified"  # type: ignore


class TestDetectBoardFormat:
    """Test auto-detection of board format from transcript content."""

    def test_default_is_standard(self) -> None:
        result = detect_board_format(False, ["Cat1"], ["Cat2"])
        assert result.name == "standard"

    def test_tournament_still_standard(self) -> None:
        result = detect_board_format(True, ["Cat1"] * 6, ["Cat2"] * 6)
        assert result.name == "standard"

    def test_classic_detected_from_100_values(self) -> None:
        # J! section has $100 but not $200
        transcript = "$100 $300 $500 " * 50 + " Double J! " + "$200 $400" * 50
        result = detect_board_format(False, ["C1"], ["C2"], transcript_text=transcript)
        assert result.name == "classic"

    def test_standard_when_200_present(self) -> None:
        # J! section has both $100 and $200 → standard (the $100 might be a score mention)
        transcript = "$200 $400 $100 $600" * 50
        result = detect_board_format(False, ["C1"], ["C2"], transcript_text=transcript)
        assert result.name == "standard"


class TestParseDollarValue:
    """Test dollar value extraction from transcript text."""

    def test_standard_dollar_sign(self) -> None:
        assert _parse_dollar_value("$800") == 800

    def test_comma_separated(self) -> None:
        assert _parse_dollar_value("$1,200") == 1200

    def test_spoken_value_for_pattern(self) -> None:
        assert _parse_dollar_value("for 600") == 600

    def test_large_value(self) -> None:
        assert _parse_dollar_value("$2,000") == 2000

    def test_no_value(self) -> None:
        assert _parse_dollar_value("no numbers here") is None

    def test_empty_string(self) -> None:
        assert _parse_dollar_value("") is None

    def test_embedded_in_sentence(self) -> None:
        assert _parse_dollar_value("Let's try Songwriters for $800 please") == 800

    def test_bare_number_rejected(self) -> None:
        """Bare numbers without $ or 'for' should NOT match (avoids Line ID false positives)."""
        assert _parse_dollar_value("line 12 of text") is None

    def test_year_not_matched(self) -> None:
        """Years like 2025 should not match without $ or 'for' prefix."""
        assert _parse_dollar_value("in the year 2025") is None


class TestInferBoardRow:
    """Test value → row mapping."""

    def test_standard_j_200(self) -> None:
        assert infer_board_row(200, "J!") == 1

    def test_standard_j_1000(self) -> None:
        assert infer_board_row(1000, "J!") == 5

    def test_standard_j_600(self) -> None:
        assert infer_board_row(600, "J!") == 3

    def test_standard_dj_400(self) -> None:
        assert infer_board_row(400, "Double J!") == 1

    def test_standard_dj_2000(self) -> None:
        assert infer_board_row(2000, "Double J!") == 5

    def test_standard_dj_1200(self) -> None:
        assert infer_board_row(1200, "Double J!") == 3

    def test_none_value_returns_middle(self) -> None:
        assert infer_board_row(None, "J!") == 3

    def test_fj_returns_row_1(self) -> None:
        assert infer_board_row(0, "Final J!") == 1

    def test_fuzzy_match_within_tolerance(self) -> None:
        # 790 should fuzzy-match to 800 (row 4 in J!)
        assert infer_board_row(790, "J!") == 4

    def test_no_match_returns_middle(self) -> None:
        # 3000 doesn't match any standard row
        assert infer_board_row(3000, "J!") == 3

    def test_celebrity_j_100(self) -> None:
        assert infer_board_row(100, "J!", CELEBRITY_BOARD) == 1

    def test_celebrity_j_500(self) -> None:
        assert infer_board_row(500, "J!", CELEBRITY_BOARD) == 5

    def test_all_standard_j_values(self) -> None:
        for i, val in enumerate(STANDARD_BOARD.j_values):
            assert infer_board_row(val, "J!") == i + 1

    def test_all_standard_dj_values(self) -> None:
        for i, val in enumerate(STANDARD_BOARD.dj_values):
            assert infer_board_row(val, "Double J!") == i + 1


class TestInferBoardRowFromSelectionText:
    """Test end-to-end selection text → row inference."""

    def test_standard_selection(self) -> None:
        assert infer_board_row_from_selection_text("Songwriters for $800", "J!") == 4

    def test_thousand_dollar_selection(self) -> None:
        assert infer_board_row_from_selection_text("I'll take it for $1,000", "J!") == 5

    def test_dj_selection(self) -> None:
        assert infer_board_row_from_selection_text("World history for $1,600", "Double J!") == 4

    def test_no_value_in_text(self) -> None:
        assert infer_board_row_from_selection_text("Pick a category", "J!") == 3


class TestRoundManifest:
    """Test round manifest construction."""

    def test_standard_j_manifest(self) -> None:
        manifest = RoundManifest(round_name="J!", categories=["A", "B", "C", "D", "E", "F"])
        assert manifest.expected_total == 30

    def test_custom_clues_per_category(self) -> None:
        manifest = RoundManifest(round_name="J!", categories=["A", "B", "C"], clues_per_category=4)
        assert manifest.expected_total == 12

    def test_empty_categories(self) -> None:
        manifest = RoundManifest(round_name="J!", categories=[])
        assert manifest.expected_total == 0


class TestBuildManifests:
    """Test manifest pair construction."""

    def test_standard_manifests(self) -> None:
        j_cats = ["A", "B", "C", "D", "E", "F"]
        dj_cats = ["G", "H", "I", "J", "K", "L"]
        j_man, dj_man = build_manifests(j_cats, dj_cats)
        assert j_man.round_name == "J!"
        assert j_man.expected_total == 30
        assert dj_man.round_name == "Double J!"
        assert dj_man.expected_total == 30

    def test_tournament_fewer_categories(self) -> None:
        j_cats = ["A", "B", "C", "D"]
        dj_cats = ["E", "F", "G", "H"]
        j_man, dj_man = build_manifests(j_cats, dj_cats)
        assert j_man.expected_total == 20
        assert dj_man.expected_total == 20

    def test_celebrity_format(self) -> None:
        j_cats = ["A", "B", "C", "D", "E", "F"]
        dj_cats = ["G", "H", "I", "J", "K", "L"]
        j_man, dj_man = build_manifests(j_cats, dj_cats, CELEBRITY_BOARD)
        assert j_man.clues_per_category == 5
        assert dj_man.clues_per_category == 5
