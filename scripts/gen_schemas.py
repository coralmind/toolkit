"""Generate JSON Schemas from the Pydantic models in ``src/corallab/schemas/``.

Schemas are written next to the YAML or JSON they validate
(``data-repository/test_case_records/schema/`` and the other data folders).
The Knowledge Library is loaded through ``coralmap`` and has no schema here.

Usage::

    pixi run gen-schemas
    pixi run check-schemas

Add a model by appending a ``SchemaTarget`` to ``REGISTRY``.
The script imports schema modules only, not simulation or judging.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from corallab.llm.registry_schema import LLMRegistry
from corallab.paths import (
    data_dir,
    eval_config_schemas_dir,
    eval_run_configs_dir,
    eval_test_cases_dir,
    repo_root,
)

# Schema modules only. Do not import simulation or judging.
from corallab.schemas.eval.persona_record import PersonaRecord
from corallab.schemas.eval.run_manifest import RunManifest
from corallab.schemas.eval.test_case_record import TestCaseRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchemaTarget:
    """One entry in the generator registry.

    Attributes:
        model: The Pydantic model class to serialise.
        output: Absolute output path for the generated JSON Schema file.
        title: Optional override for the schema's top-level ``title``.
        post_process: Optional hook that mutates the schema dict in place
            (e.g. inject runtime ontology enum values). Called after
            ``model.model_json_schema()`` and before writing to disk.
    """

    model: type[BaseModel]
    output: Path
    title: str | None = None
    post_process: Callable[[dict[str, Any]], None] | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Post-processing hooks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: list[SchemaTarget] = [
    # --- Run manifests (run_manifests/schema/) ---
    SchemaTarget(
        RunManifest,
        eval_run_configs_dir() / "schema" / "run_manifest.schema.json",
        title="Run Manifest",
    ),
    # --- Persona and test case records (persona_records/schema/, test_case_records/schema/) ---
    SchemaTarget(
        PersonaRecord,
        data_dir() / "persona_records" / "schema" / "persona_record.schema.json",
        title="Persona Record",
    ),
    SchemaTarget(
        TestCaseRecord,
        eval_test_cases_dir() / "schema" / "test_case_record.schema.json",
        title="Test Case Record",
    ),
    # --- LLM endpoint registry (endpoints/schema/) ---
    SchemaTarget(
        LLMRegistry,
        eval_config_schemas_dir() / "endpoints.schema.json",
        title="LLM Endpoint Registry",
    ),
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _render(target: SchemaTarget) -> str:
    """Render one schema to its canonical JSON string (trailing newline included)."""
    schema = target.model.model_json_schema(
        ref_template="#/$defs/{model}",
        mode="serialization",
    )
    if target.title is not None:
        schema["title"] = target.title
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    if target.post_process is not None:
        target.post_process(schema)
    return json.dumps(schema, indent=2, sort_keys=False) + "\n"


def _write(target: SchemaTarget) -> None:
    """Generate ``target`` and write it to disk (creating parent dirs)."""
    output_path = target.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render(target), encoding="utf-8")
    logger.info("wrote %s", output_path.relative_to(repo_root()))


def _check(target: SchemaTarget) -> bool:
    """Return True if the on-disk file matches the freshly rendered schema."""
    output_path = target.output
    rendered = _render(target)
    if not output_path.exists():
        logger.error("missing: %s", output_path.relative_to(repo_root()))
        return False
    on_disk = output_path.read_text(encoding="utf-8")
    if on_disk != rendered:
        logger.error("drift: %s", output_path.relative_to(repo_root()))
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify on-disk schemas match the Pydantic models; exit non-zero on drift.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.check:
        ok = all(_check(t) for t in REGISTRY)
        if not ok:
            logger.error("Schemas are out of date. Run `pixi run gen-schemas` to regenerate.")
            return 1
        logger.info("all %d schemas up to date.", len(REGISTRY))
        return 0

    for target in REGISTRY:
        _write(target)
    logger.info("generated %d schemas.", len(REGISTRY))
    return 0


if __name__ == "__main__":
    sys.exit(main())
