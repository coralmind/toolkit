"""Tests for ``corallab.cli.check_env``.

Synthesizes a small ``endpoints.yaml`` and asserts referenced var names are
scraped, presence (not values) is reported, and a missing ``.env`` exits
non-zero. No LLM calls; env vars are monkeypatched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from corallab.cli import app
from corallab.cli.check_env import referenced_vars

_REGISTRY_YAML = """\
endpoints:
    azure-default:
        provider: azure
        api_base: ${AZURE_API_BASE}
        api_key_env: AZURE_API_KEY
    proxy:
        provider: openai
        api_base: ${LITELLM_API_BASE}
        api_key_env: LITELLM_API_KEY
    no-auth-local:
        provider: openai
        api_base: http://localhost:11434/v1
        api_key_env: null
"""


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    path = tmp_path / "endpoints.yaml"
    path.write_text(_REGISTRY_YAML)
    return path


class TestReferencedVars:
    def test_scrapes_api_key_env_and_interpolation_tokens(self, registry_path: Path) -> None:
        names = referenced_vars(registry_path)
        assert {"AZURE_API_KEY", "AZURE_API_BASE", "LITELLM_API_KEY", "LITELLM_API_BASE"} <= set(
            names
        )

    def test_null_api_key_env_and_literal_base_add_nothing(self, registry_path: Path) -> None:
        names = referenced_vars(registry_path)
        assert all("localhost" not in n for n in names)

    def test_missing_registry_references_nothing(self, tmp_path: Path) -> None:
        assert referenced_vars(tmp_path / "missing.yaml") == []

    def test_returns_sorted(self, registry_path: Path) -> None:
        names = referenced_vars(registry_path)
        assert names == sorted(names)


class TestCli:
    def test_reports_presence_without_printing_values(
        self, registry_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "super-secret-should-never-print"
        monkeypatch.setenv("AZURE_API_KEY", secret)
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        # Pretend a .env was located so the result is independent of whether the
        # machine running the test actually has one (CI does not; dev does).
        monkeypatch.setattr("corallab.cli.check_env.find_dotenv", lambda: "/tmp/.env")
        result = CliRunner().invoke(app, ["check-env", "--registry", str(registry_path)])
        assert result.exit_code == 0, result.output
        assert "AZURE_API_KEY" in result.output
        assert secret not in result.output

    def test_exits_nonzero_when_no_dotenv(
        self, registry_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("corallab.cli.check_env.find_dotenv", lambda: "")
        result = CliRunner().invoke(app, ["check-env", "--registry", str(registry_path)])
        assert result.exit_code == 1
