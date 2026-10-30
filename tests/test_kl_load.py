"""The toolkit can load a Knowledge Library through ``CORALMIND_KL_PATH``."""

from __future__ import annotations

from pathlib import Path

import pytest
from coralmap import load
from coralmap.loader import clear_cache

from corallab.paths import KL_PATH_ENV_VAR


def test_load_synthetic_library_via_env(
    coralmap_fixture_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KL_PATH_ENV_VAR, str(coralmap_fixture_path))
    clear_cache()
    library = load()
    assert library.version
    assert library.digest.startswith("sha256:")
    assert {entry.id for entry in library.behaviors} == {"kl-900001", "kl-900002"}
    assert library.entry("kl-900101").display_id == "Pathway-901"
