"""Round-trip tests for ``scripts/gen_schemas.py``.

The REGISTRY targets (run manifest, persona record, test case record, LLM registry) must
match the committed JSON Schemas.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from gen_schemas import REGISTRY, SchemaTarget, _render, main

from corallab.paths import repo_root

_DATA_FILES: dict[str, list[Path]] = {
    # Run manifests and the endpoint registry are validated by their own tests.
    "RunManifest": [],
    "LLMRegistry": [],
}


def _id(target: SchemaTarget) -> str:
    return target.model.__name__


@pytest.mark.parametrize("target", REGISTRY, ids=_id)
def test_committed_schema_matches_pydantic_model(target: SchemaTarget) -> None:
    """The on-disk schema file matches what the generator emits today."""
    output_path = target.output
    assert output_path.exists(), (
        f"Missing schema file {output_path.relative_to(repo_root())}. Run `pixi run gen-schemas`."
    )

    expected = _render(target)
    on_disk = output_path.read_text(encoding="utf-8")
    assert on_disk == expected, (
        f"Schema drift in {target.output.relative_to(repo_root())} — Pydantic model has changed "
        "but the committed schema has not been regenerated. Run `pixi run gen-schemas`."
    )

    json.loads(on_disk)


@pytest.mark.parametrize("target", REGISTRY, ids=_id)
def test_authored_files_validate_against_pydantic_model(target: SchemaTarget) -> None:
    """Every committed authored file matching this target validates cleanly."""
    files = _DATA_FILES.get(target.model.__name__)
    if not files:
        pytest.skip(f"No authored files registered for {target.model.__name__}")
    for abs_path in files:
        if not abs_path.exists():
            pytest.skip(f"Authored file missing on disk: {abs_path}")
        raw = yaml.safe_load(abs_path.read_text(encoding="utf-8"))
        target.model.model_validate(raw)


def test_check_mode_passes() -> None:
    """End-to-end smoke: ``python scripts/gen_schemas.py --check`` returns 0."""
    assert main(["--check"]) == 0
