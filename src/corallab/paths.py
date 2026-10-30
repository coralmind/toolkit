"""Filesystem path resolution for CoralMind toolkit data and config files.

``repo_root()`` and ``data_dir()`` are the entry points; the other helpers compose off
those roots.
``repo_root()`` honors ``CORALMIND_TOOLKIT_ROOT``. ``data_dir()`` honors
``CORALMIND_DATA_REPOSITORY_PATH`` — see ``data_dir()`` for the full resolution order.

Path resolution is lazy (function calls, not module-level constants) so that
tests can ``monkeypatch.setenv("CORALMIND_TOOLKIT_ROOT", str(tmp_path))`` without
worrying about import order.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT_ENV_VAR = "CORALMIND_TOOLKIT_ROOT"
DATA_REPOSITORY_PATH_ENV_VAR = "CORALMIND_DATA_REPOSITORY_PATH"
# Owned by coralmap; recorded here so toolkit code does not invent a second name.
KL_PATH_ENV_VAR = "CORALMIND_KL_PATH"


def default_repo_root() -> Path:
    """Return the repo root derived from the package layout.

    Ignores ``CORALMIND_TOOLKIT_ROOT``. Use in test fixtures that must symlink
    real checkout data into a temp repo without honoring the override.
    """
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """Return the toolkit repository root.

    Honors ``CORALMIND_TOOLKIT_ROOT`` if set; otherwise derives from the package
    install location (two parents up from this file: ``corallab/`` →
    ``src/`` → repo root).
    """
    override = os.environ.get(REPO_ROOT_ENV_VAR)
    if override:
        return Path(override)
    return default_repo_root()


def repo_relative(path: Path) -> Path:
    """Return ``path`` relative to ``repo_root()`` when it lives under the repo."""
    return path.relative_to(repo_root())


def package_dir() -> Path:
    """Return the ``corallab`` package directory."""
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    """Return the directory holding generated and specification artifacts.

    Resolution order:

    1. ``CORALMIND_DATA_REPOSITORY_PATH`` if set.
    2. Package-bundled data: ``<package>/data-repository`` when that directory
       exists. Nothing ships one today.
    3. Checkout fallback: ``repo_root() / "data-repository"``.
    """
    override = os.environ.get(DATA_REPOSITORY_PATH_ENV_VAR)
    if override:
        return Path(override)

    bundled = package_dir() / "data-repository"
    if bundled.is_dir():
        return bundled

    return repo_root() / "data-repository"


#: A path may name this directory explicitly. That leading segment is dropped.
DATA_ROOT_PREFIX = "data-repository"


def resolve_data_path(value: str | Path) -> Path:
    """Resolve a manifest-supplied path against the data repository.

    Resolution order:

    1. An absolute path is returned unchanged.
    2. A leading ``data-repository/`` segment is dropped and the remainder
       resolved under :func:`data_dir`, even when that file is absent.
    3. A relative path that exists under :func:`data_dir` resolves there.
       When the same relative path also exists in the working directory, the
       data root wins.
    4. Anything else is returned unchanged.
    """
    path = Path(value)
    if path.is_absolute():
        return path

    parts = path.parts
    if parts and parts[0] == DATA_ROOT_PREFIX:
        return data_dir().joinpath(*parts[1:])

    candidate = data_dir() / path
    if candidate.exists():
        return candidate

    return path


def resolve_run_dir(value: str | Path) -> Path:
    """Place a manifest ``run_dir`` under the data repository.

    An absolute path is returned unchanged. A leading ``data-repository/``
    segment is dropped. Any other relative path is
    placed under :func:`data_dir` even when the directory does not exist
    yet, so ``results/foo`` is ``<data-repository>/results/foo``.
    """
    path = Path(value)
    if path.is_absolute():
        return path

    parts = path.parts
    if parts and parts[0] == DATA_ROOT_PREFIX:
        path = Path(*parts[1:]) if len(parts) > 1 else Path()
    return data_dir() / path


def knowledge_library_path() -> Path | None:
    """Return the ``CORALMIND_KL_PATH`` override, if set.

    Full resolution (override, then the wheel bundle) lives in
    ``coralmap.load``. The toolkit does not reimplement that order.
    """
    override = os.environ.get(KL_PATH_ENV_VAR)
    return Path(override) if override else None


def eval_run_configs_dir() -> Path:
    """Return ``run_manifests/`` — run manifest YAML files."""
    return data_dir() / "run_manifests"


def eval_test_cases_dir() -> Path:
    """Return ``test_case_records/`` — authored test-case JSONL files."""
    return data_dir() / "test_case_records"


def eval_config_schemas_dir() -> Path:
    """Return ``endpoints/schema/`` — the endpoint registry schema."""
    return data_dir() / "endpoints" / "schema"


def eval_persona_pool_dir() -> Path:
    """Return ``persona_records/`` — persona record JSON files (``p-*.json``)."""
    return data_dir() / "persona_records"


def endpoints_registry_path() -> Path:
    """Return ``endpoints/endpoints.yaml`` — the LLM endpoint registry."""
    return data_dir() / "endpoints" / "endpoints.yaml"


def results_dir() -> Path:
    """Return ``results/`` under the data repository — ephemeral run outputs."""
    return data_dir() / "results"


def judge_assets_dir() -> Path:
    """Return ``src/corallab/judging/judges/prompts/`` — versioned judge prompts."""
    return package_dir() / "judging" / "judges" / "prompts"
