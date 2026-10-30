"""Tests for CLI helpers — command name extraction and run stem derivation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from corallab.cli import _helpers
from corallab.cli._helpers import (
    LogLevel,
    init_logging,
    make_run_stem,
    print_token_summary,
    safe_name,
    transcript_run_dir,
    unique_name,
)
from corallab.llm_client import TokenAccumulator
from corallab.schemas.eval.run_manifest import RunManifest

_JUDGE = {
    "judge_type": "kl_two_stage",
    "model": {"model_id": "test-judge"},
}

runner = CliRunner()


def test_render_group_is_gone() -> None:
    assert not hasattr(_helpers, "render_group")


@pytest.fixture
def manifest() -> RunManifest:
    return RunManifest(
        run_id="test-001",
        evaluation_level=1,
        system_under_test=[{"model_id": "test-sut"}],
        test_case_records_path="input.jsonl",
        run_dir="my_results",
        judges=[_JUDGE],
    )


class TestMakeRunStem:
    """make_run_stem names folders for commands that have no run_dir."""

    def test_uses_command_name_and_timestamp(self) -> None:
        stem = make_run_stem("judge")
        assert stem.startswith("corallab-judge-")

    def test_run_command_stem(self) -> None:
        stem = make_run_stem("run")
        assert stem.startswith("corallab-run-")


class TestInitLoggingCommandName:
    """init_logging extracts the command name from ctx.info_name."""

    @pytest.mark.parametrize(
        ("info_name", "expected_prefix"),
        [
            ("sim", "corallab-sim-"),
            ("judge", "corallab-judge-"),
            ("run", "corallab-run-"),
            ("run-preview", "corallab-run-preview-"),
        ],
    )
    def test_run_stem_uses_ctx_info_name(
        self,
        info_name: str,
        expected_prefix: str,
        tmp_path,
    ) -> None:
        ctx = MagicMock(spec=typer.Context)
        ctx.info_name = info_name
        ctx.obj = {"log_level": LogLevel.WARNING}

        with patch("corallab.cli._helpers.results_dir", return_value=tmp_path):
            _out_dir, run_stem = init_logging(ctx)

        assert run_stem.startswith(expected_prefix)

    def test_manifest_run_dir_is_timestamped_so_reruns_do_not_overwrite(
        self,
        manifest: RunManifest,
        tmp_path: Path,
    ) -> None:
        ctx = MagicMock(spec=typer.Context)
        ctx.info_name = "sim"
        ctx.obj = {"log_level": LogLevel.WARNING}

        manifest.run_dir = str(tmp_path / "my_results")
        fixed = datetime(2026, 10, 7, 15, 47, 0)
        with patch("corallab.cli._helpers.datetime") as clock:
            clock.now.return_value = fixed
            out_dir, run_stem = init_logging(ctx, manifest)

        expected = tmp_path / "my_results-20261007-154700"
        assert out_dir == expected
        assert run_stem == "my_results-20261007-154700"
        assert expected.is_dir()
        assert not (tmp_path / "my_results").exists()

    def test_same_second_manifest_run_gets_a_distinct_directory(
        self,
        manifest: RunManifest,
        tmp_path: Path,
    ) -> None:
        ctx = MagicMock(spec=typer.Context)
        ctx.info_name = "sim"
        ctx.obj = {"log_level": LogLevel.WARNING}
        manifest.run_dir = str(tmp_path / "my_results")
        (tmp_path / "my_results-20261007-154700").mkdir()

        fixed = datetime(2026, 10, 7, 15, 47, 0)
        with patch("corallab.cli._helpers.datetime") as clock:
            clock.now.return_value = fixed
            out_dir, run_stem = init_logging(ctx, manifest)

        assert out_dir == tmp_path / "my_results-20261007-154700-2"
        assert run_stem == "my_results-20261007-154700-2"
        assert out_dir.is_dir()

    def test_explicit_out_dir_is_not_timestamped(
        self,
        manifest: RunManifest,
        tmp_path: Path,
    ) -> None:
        ctx = MagicMock(spec=typer.Context)
        ctx.info_name = "judge"
        ctx.obj = {"log_level": LogLevel.WARNING}
        manifest.run_dir = str(tmp_path / "my_results")
        existing = tmp_path / "my_results-20261007-154700"
        existing.mkdir()

        out_dir, run_stem = init_logging(ctx, manifest, out_dir=existing)

        assert out_dir == existing
        assert run_stem == existing.name

    def test_none_info_name_falls_back_to_corallab(self, tmp_path) -> None:
        ctx = MagicMock(spec=typer.Context)
        ctx.info_name = None
        ctx.obj = {"log_level": LogLevel.WARNING}

        with patch("corallab.cli._helpers.results_dir", return_value=tmp_path):
            _out_dir, run_stem = init_logging(ctx)

        assert run_stem.startswith("corallab-corallab-")


class TestPrintTokenSummary:
    """print_token_summary renders the accumulator's summary to the CLI console."""

    def test_prints_summary_to_console(self, capsys) -> None:
        acc = TokenAccumulator()
        acc.record(
            role="judge", model="claude-x", input_tokens=1234, output_tokens=56, cache_tokens=7
        )
        print_token_summary(acc, "judge")

        out = capsys.readouterr().out
        assert "Token usage — judge" in out
        assert "1,234" in out
        assert "TOTAL" in out

    def test_empty_accumulator_does_not_raise(self, capsys) -> None:
        print_token_summary(TokenAccumulator(), "sim")
        assert "TOTAL" in capsys.readouterr().out


class TestCliCommandNames:
    """Verify that Typer registers commands with the expected info_name values."""

    @pytest.mark.parametrize(
        "command_name",
        ["sim", "judge", "run", "run-preview"],
    )
    def test_help_invocation_succeeds(self, command_name: str) -> None:
        from corallab.cli import app

        result = runner.invoke(app, [command_name, "--help"])
        assert result.exit_code == 0
        assert command_name in result.output.lower() or "usage" in result.output.lower()


# --- safe_name / unique_name (output path component sanitizers) -------------


def test_safe_name_passes_clean_ids_unchanged() -> None:
    assert safe_name("judge-3f1aa08e") == "judge-3f1aa08e"
    assert safe_name("aimhef-31476d5c") == "aimhef-31476d5c"


def test_safe_name_collapses_separators_and_specials() -> None:
    assert safe_name("vendor/model 1") == "vendor_model_1"
    assert safe_name("../../x") == ".._.._x"


def test_safe_name_rejects_dot_only_and_empty() -> None:
    # These would resolve to current/parent dir as a path component.
    assert safe_name("..") == "unknown"
    assert safe_name(".") == "unknown"
    assert safe_name("...") == "unknown"
    assert safe_name("") == "unknown"


def test_transcript_run_dir_is_the_batch_parent(tmp_path: Path) -> None:
    run = tmp_path / "my_results-20261007-154700"
    batch = run / "batch-001"
    batch.mkdir(parents=True)
    transcripts = batch / "my_results-20261007-154700-sim.jsonl"
    transcripts.write_text("{}\n")

    assert transcript_run_dir(transcripts) == run


def test_transcript_run_dir_uses_the_parent_when_the_file_is_not_in_a_batch(
    tmp_path: Path,
) -> None:
    run = tmp_path / "my_results-20261007-154700"
    run.mkdir()
    transcripts = run / "my_results-20261007-154700-sim.jsonl"
    transcripts.write_text("{}\n")

    assert transcript_run_dir(transcripts) == run


def test_transcript_run_dir_is_none_when_the_file_is_missing(tmp_path: Path) -> None:
    assert transcript_run_dir(tmp_path / "missing.jsonl") is None


def test_unique_name_suffixes_repeated_collisions() -> None:
    seen: set[str] = set()
    # Same base four times must yield four distinct names — the bug was that
    # the 2nd and 3rd both mapped to "x-1".
    names = [unique_name("x", seen) for _ in range(4)]
    assert names == ["x", "x-2", "x-3", "x-4"]
    assert len(set(names)) == 4
