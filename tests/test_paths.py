"""Tests for ``corallab.paths`` resolution helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from corallab import paths


def test_repo_root_honors_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.REPO_ROOT_ENV_VAR, str(tmp_path))
    assert paths.repo_root() == tmp_path
    assert paths.data_dir() == tmp_path / "data-repository"


def test_data_dir_honors_coralmind_data_repository_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "elsewhere"
    monkeypatch.setenv(paths.DATA_REPOSITORY_PATH_ENV_VAR, str(data))
    monkeypatch.setenv(paths.REPO_ROOT_ENV_VAR, str(tmp_path / "repo"))
    assert paths.data_dir() == data


def test_eval_dirs_split_by_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """test_case_records/ run_manifests/ endpoints/ replace the old eval/configs/ grouping.

    It held three unlike things: test cases (data), run manifests, and an
    endpoint registry (environment routing).
    """
    monkeypatch.setenv(paths.REPO_ROOT_ENV_VAR, str(tmp_path))
    root = tmp_path / "data-repository"
    assert paths.eval_test_cases_dir() == root / "test_case_records"
    assert paths.eval_run_configs_dir() == root / "run_manifests"
    assert paths.eval_persona_pool_dir() == root / "persona_records"
    assert paths.endpoints_registry_path() == root / "endpoints" / "endpoints.yaml"
    assert paths.eval_config_schemas_dir() == root / "endpoints" / "schema"


def test_data_runs_dir_is_gone() -> None:
    """It had no callers, and a ``runs/`` directory under the data root is not a path code
    should compose.

    Repointing an unused helper at the new layout would have preserved dead
    code through a rename, which is the moment it is cheapest to drop.
    """
    assert not hasattr(paths, "data_runs_dir")


def test_docs_site_path_helpers_are_gone() -> None:
    assert not hasattr(paths, "websites_dir")
    assert not hasattr(paths, "docs_content_dir")
    assert not hasattr(paths, "docs_api_reference_dir")


def test_unused_repo_path_helpers_are_gone() -> None:
    assert not hasattr(paths, "resolve_repo_path")
    assert not hasattr(paths, "src_dir")
    assert not hasattr(paths, "temp_dir")
    assert not hasattr(paths, "tests_dir")
    assert not hasattr(paths, "tests_fixtures_dir")


def test_repo_relative_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.REPO_ROOT_ENV_VAR, str(tmp_path))
    nested = paths.eval_test_cases_dir() / "alpha" / "cases.jsonl"
    assert paths.repo_relative(nested) == Path(
        "data-repository/test_case_records/alpha/cases.jsonl"
    )


def test_knowledge_library_path_delegates_to_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(paths.KL_PATH_ENV_VAR, raising=False)
    assert paths.knowledge_library_path() is None
    monkeypatch.setenv(paths.KL_PATH_ENV_VAR, str(tmp_path))
    assert paths.knowledge_library_path() == tmp_path


def test_package_paths_are_under_package_dir() -> None:
    pkg = paths.package_dir()
    assert paths.judge_assets_dir().is_relative_to(pkg)


def test_resolve_data_path_passes_absolute_through(tmp_path: Path) -> None:
    """An absolute path comes back byte-for-byte identical.

    This pins the contract, not the ``is_absolute()`` branch: pathlib join
    semantics make the later branches return the same answer, so deleting the
    guard would not turn this red. What it does catch is an implementation
    that normalises the path — ``.resolve()``, ``.expanduser()``, or rebasing
    on ``path.name``.
    """
    absolute = tmp_path / "somewhere" / "cases.jsonl"
    assert paths.resolve_data_path(absolute) == absolute


def test_resolve_data_path_drops_a_leading_data_repository_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.DATA_REPOSITORY_PATH_ENV_VAR, str(tmp_path / "data-repository"))

    resolved = paths.resolve_data_path("data-repository/test_case_records/l1.jsonl")

    assert resolved == tmp_path / "data-repository" / "test_case_records/l1.jsonl"


def test_resolve_data_path_leaves_a_leading_data_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``data/`` is an ordinary relative path, not a root prefix."""
    monkeypatch.setenv(paths.DATA_REPOSITORY_PATH_ENV_VAR, str(tmp_path / "data-repository"))

    assert paths.resolve_data_path("data/test_case_records/l1.jsonl") == Path(
        "data/test_case_records/l1.jsonl"
    )


def test_resolve_run_dir_drops_only_the_data_repository_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.DATA_REPOSITORY_PATH_ENV_VAR, str(tmp_path / "data-repository"))
    root = tmp_path / "data-repository"

    assert paths.resolve_run_dir("data-repository/results/foo") == root / "results/foo"
    assert paths.resolve_run_dir("data/results/foo") == root / "data/results/foo"


def test_resolve_data_path_finds_a_bare_relative_path_under_the_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spelling new manifests use: data-root-relative, no prefix."""
    root = tmp_path / "data-repository"
    (root / "test_case_records").mkdir(parents=True)
    (root / "test_case_records" / "l1.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv(paths.DATA_REPOSITORY_PATH_ENV_VAR, str(root))

    assert (
        paths.resolve_data_path("test_case_records/l1.jsonl")
        == root / "test_case_records" / "l1.jsonl"
    )


def test_resolve_data_path_prefers_the_data_root_over_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a bare relative path exists in both places, the data root wins."""
    root = tmp_path / "data-repository"
    (root / "test_case_records").mkdir(parents=True)
    (root / "test_case_records" / "l1.jsonl").write_text("from the data root\n", encoding="utf-8")

    cwd = tmp_path / "cwd"
    (cwd / "test_case_records").mkdir(parents=True)
    (cwd / "test_case_records" / "l1.jsonl").write_text(
        "from the working directory\n", encoding="utf-8"
    )

    monkeypatch.setenv(paths.DATA_REPOSITORY_PATH_ENV_VAR, str(root))
    monkeypatch.chdir(cwd)

    assert (
        paths.resolve_data_path("test_case_records/l1.jsonl")
        == root / "test_case_records" / "l1.jsonl"
    )


def test_resolve_data_path_leaves_an_unmatched_relative_path_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path that is not under the data root is returned unchanged."""
    monkeypatch.setenv(paths.DATA_REPOSITORY_PATH_ENV_VAR, str(tmp_path / "data-repository"))

    assert paths.resolve_data_path("scratch/x.jsonl") == Path("scratch/x.jsonl")


def test_no_source_or_config_file_references_the_old_data_layout() -> None:
    """Nothing still points at the old data tree.

    ``simulation/data/`` contains ``data`` for an unrelated reason, so the
    pattern requires one of the old child directories after the root.
    ``data-repository/`` followed by a current child is not a match.
    """
    import re

    root = paths.repo_root()
    # Pin the root and the scan size. An empty offender list is also what a
    # misresolved repo_root() would produce.
    assert (root / "AGENTS.md").exists(), f"repo_root() resolved to {root}, nothing was scanned"

    searched = [
        *(root / "src").rglob("*.py"),
        *(root / "scripts").rglob("*.py"),
        *(root / "tests").rglob("*.py"),
        root / "README.md",
        root / "AGENTS.md",
        root / ".env.example",
        root / "pyproject.toml",
        *(root / ".github" / "workflows").glob("*.yml"),
        # Authored YAML in the data repository itself — endpoints.yaml carried a
        # stale path in its own header comment, which nothing outside the tree
        # could have caught.
        *(root / "data-repository").rglob("*.yaml"),
    ]
    assert len(searched) > 100, f"only {len(searched)} files in scope — the globs found too little"

    stale = re.compile(r"(?<![\w-])data(-repository)?/(bench|eval|runs|knowledge)\b")

    offenders = [
        f"{path.relative_to(root)}:{n}: {line.strip()}"
        for path in searched
        if path.exists()
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if stale.search(line)
    ]

    assert not offenders, "stale data/ references:\n" + "\n".join(offenders)
