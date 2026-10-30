"""The toolkit's policy for loading the Knowledge Library: whole, or not at all."""

from __future__ import annotations

import pathlib
import shutil

import pytest
from coralmap import LibraryParseError
from pydantic import ValidationError

from corallab import kl
from corallab.schemas.core.kl import KlPin


def _fixture_copy(source: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    """Copy the synthetic library so a test can damage it without damaging the tree."""
    target = tmp_path / "kl"
    shutil.copytree(source, target)
    return target


def test_load_library_loads_a_clean_library(
    coralmap_fixture_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    library = kl.load_library(_fixture_copy(coralmap_fixture_path, tmp_path))

    assert library.all_entries()


def test_the_toolkit_names_no_entry_ids_of_its_own() -> None:
    """An entry id hardcoded here would be valid for exactly one library."""
    source = pathlib.Path(kl.__file__).read_text(encoding="utf-8")

    assert "kl-0" not in source, "an entry id is hardcoded in the toolkit"


def test_a_broken_entry_stops_the_load(
    coralmap_fixture_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A run never proceeds on part of a library."""
    root = _fixture_copy(coralmap_fixture_path, tmp_path)
    broken = root / "entries" / "ai_behaviors" / "kl-900002.yaml"
    broken.write_text("id: kl-900002\ntype: ai_behavior\n", encoding="utf-8")

    with pytest.raises(LibraryParseError) as caught:
        kl.load_library(root)

    assert "kl-900002" in str(caught.value)


def test_kl_pin_records_version_and_digest(
    coralmap_fixture_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    library = kl.load_library(_fixture_copy(coralmap_fixture_path, tmp_path))
    pin = KlPin.from_library(library)

    assert pin.version == library.version
    assert pin.digest == library.digest


def test_a_pin_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        KlPin.model_validate({"version": "0.2.0", "digest": "sha256:0", "extra": True})
