"""Resolve the package versions of the code that is writing a record."""

from __future__ import annotations

import importlib.metadata
import logging
import subprocess
from functools import lru_cache
from pathlib import Path

import coralmap

from corallab.schemas.core.code_version import PackageVersion, ProducedBy

logger = logging.getLogger(__name__)

_TOOLKIT = "corallab"
_KNOWLEDGE_LIBRARY = "coralmap"


def produced_by() -> ProducedBy:
    """Versions of the toolkit and Knowledge Library in this process.

    Cached for the process. A missing install or a non-checkout reports a
    version of ``unknown`` and a null revision rather than failing the run.
    """
    return ProducedBy(
        toolkit=_package(_TOOLKIT, Path(__file__)),
        knowledge_library=_package(
            _KNOWLEDGE_LIBRARY, Path(coralmap.__file__), coralmap.__version__
        ),
    )


def _package(name: str, source: Path, fallback_version: str | None = None) -> PackageVersion:
    revision, dirty = _git_revision(source)
    return PackageVersion(
        name=name,
        version=_dist_version(name, fallback_version),
        revision=revision,
        dirty=dirty,
    )


def _dist_version(name: str, fallback: str | None) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        if fallback:
            return fallback
        logger.debug("Package %s is not installed; recording version as unknown", name)
        return "unknown"


@lru_cache(maxsize=8)
def _git_revision(source: Path) -> tuple[str | None, bool | None]:
    """Return ``(commit, dirty)`` for the checkout that contains ``source``.

    A wheel under site-packages is not a checkout. ``git -C`` would otherwise
    walk up into a repository that merely contains the environment and record
    that other commit. The pixi env lives inside this repo, so a PyPI
    ``coralmap`` would be stamped with the toolkit's revision.
    """
    directory = source if source.is_dir() else source.parent
    if _is_installed_wheel(directory):
        return None, None
    commit = _git(directory, "rev-parse", "HEAD")
    if commit is None:
        return None, None
    status = _git(directory, "status", "--porcelain")
    if status is None:
        return commit, None
    return commit, bool(status)


def _is_installed_wheel(directory: Path) -> bool:
    parts = directory.resolve().parts
    return "site-packages" in parts or "dist-packages" in parts


def _git(directory: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("git %s failed for %s", args[0], directory, exc_info=True)
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
