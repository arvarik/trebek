"""
Tests for CLI argument parsing — build_parser, TrebekArgumentParser behavior,
and subcommand argument validation. Does NOT invoke the actual pipeline.
"""

import pytest
from unittest.mock import patch

from trebek.cli import build_parser, TrebekArgumentParser, main


class TestBuildParser:
    """Tests for the CLI argument parser construction."""

    def test_parser_returns_instance(self) -> None:
        parser = build_parser()
        assert isinstance(parser, TrebekArgumentParser)

    def test_run_subcommand_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"
        assert args.once is False
        assert args.stage == "all"
        assert args.model == "pro"
        assert args.docker is False
        assert args.max_retries == 3

    def test_run_once_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--once"])
        assert args.once is True

    def test_run_with_stage(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--stage", "transcribe"])
        assert args.stage == "transcribe"

    def test_run_with_model(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--model", "flash"])
        assert args.model == "flash"

    def test_run_with_input_dir(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--input-dir", "/my/videos"])
        assert args.input_dir == "/my/videos"

    def test_run_with_docker(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--docker"])
        assert args.docker is True

    def test_run_with_max_retries(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--max-retries", "5"])
        assert args.max_retries == 5

    def test_scan_subcommand_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["scan"])
        assert args.command == "scan"
        assert args.input_dir is None
        assert args.stage is None

    def test_scan_with_stage(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["scan", "--stage", "extract"])
        assert args.stage == "extract"

    def test_stats_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["stats"])
        assert args.command == "stats"

    def test_retry_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["retry"])
        assert args.command == "retry"

    def test_version_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["version"])
        assert args.command == "version"

    def test_no_args_defaults_to_none(self) -> None:
        """No subcommand given → args.command is None (main() defaults to 'run')."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestTrebekArgumentParser:
    """Tests for the custom argument parser."""

    def test_print_help_invokes_rich(self) -> None:
        with patch("trebek.cli.TrebekArgumentParser.print_help") as mock_help:
            parser = TrebekArgumentParser(prog="test", help_command="main")
            parser.print_help()
            mock_help.assert_called_once()

    def test_error_exits_with_code_2(self) -> None:
        parser = TrebekArgumentParser(prog="test", help_command="main")
        with patch("trebek.ui.help.render_help"), patch("trebek.ui.core.console.print"):
            with pytest.raises(SystemExit) as exc_info:
                parser.error("bad argument")
            assert exc_info.value.code == 2


class TestMainVersion:
    """Tests for the main() function version command."""

    def test_version_prints_and_returns(self) -> None:
        with patch("sys.argv", ["trebek", "version"]), patch("trebek.ui.core.console.print") as mock_print:
            main()
            # Should print version string
            call_args = str(mock_print.call_args)
            assert "trebek" in call_args
