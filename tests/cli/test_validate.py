"""``corallab validate test-cases`` and ``validate run-manifest``."""

from __future__ import annotations

from pathlib import Path

import pytest
from _records import make_persona, make_test_case, observed_fixture_copy, write_test_case_records
from click.testing import Result
from typer.testing import CliRunner

from corallab.cli import app

runner = CliRunner()


@pytest.fixture
def observed_kl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI's ``load_library`` at a fixture copy whose pathway observes kl-900501.

    Records must be built after this runs: their library pin is read from the same copy.
    """
    root = observed_fixture_copy(tmp_path / "kl")
    monkeypatch.setenv("CORALMIND_KL_PATH", str(root))
    return root


def _output(result: Result) -> str:
    return result.output + (result.stderr if result.stderr_bytes is not None else "")


def test_good_test_case_records_validate(tmp_path: Path, observed_kl: Path) -> None:
    records_path, personas = write_test_case_records(tmp_path, [make_test_case()], [make_persona()])
    result = runner.invoke(
        app, ["validate", "test-cases", str(records_path), "--personas", str(personas)]
    )
    assert result.exit_code == 0, _output(result)
    text = " ".join(_output(result).split())
    assert "OK" in text
    assert "1 test case" in text


def test_a_missing_persona_is_reported(tmp_path: Path, observed_kl: Path) -> None:
    records_path, personas = write_test_case_records(tmp_path, [make_test_case()], [])
    result = runner.invoke(
        app, ["validate", "test-cases", str(records_path), "--personas", str(personas)]
    )
    assert result.exit_code == 1
    assert "p-fix-01@1" in _output(result)


def test_a_wrong_assertion_is_reported(tmp_path: Path, observed_kl: Path) -> None:
    records_path, personas = write_test_case_records(
        tmp_path, [make_test_case(pathway_id="kl-900101")], [make_persona()]
    )
    result = runner.invoke(
        app, ["validate", "test-cases", str(records_path), "--personas", str(personas)]
    )
    assert result.exit_code == 1
    assert "its pathway expects" in _output(result)


def test_a_run_manifest_validates(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        """\
# yaml-language-server: $schema=./schema/run_manifest.schema.json
run_id: r
evaluation_level: 1
system_under_test:
- model_id: s
test_case_records_path: test_case_records/examples/l1_launch_smoke.jsonl
run_dir: results/r
judges:
- judge_type: kl_two_stage
  model:
      model_id: j
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", "run-manifest", str(path)])
    assert result.exit_code == 0, _output(result)


def test_a_bad_run_manifest_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("run_id: r\nevaluation_level: 3\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", "run-manifest", str(path)])
    assert result.exit_code == 1
    assert "system_under_test" in _output(result)
    assert "Field required" in _output(result)
    assert "input_value" in _output(result)


def test_missing_test_case_records_are_reported(tmp_path: Path, observed_kl: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    result = runner.invoke(
        app, ["validate", "test-cases", str(missing), "--personas", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "nope.jsonl" in _output(result)
    assert isinstance(result.exception, SystemExit)


def test_a_missing_run_manifest_is_reported(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", "run-manifest", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1
    assert "nope.yaml" in _output(result)
    assert isinstance(result.exception, SystemExit)
