"""Copy a finished run into ``data-repository/sample_results/`` for the public site.

Only the files the pages read are copied: each batch's sim, check and judge JSONL and their
trace files, and ``inputs/``. Logs and Burr databases are never copied. The persona records the
run's test cases name are copied into ``inputs/persona_records/``, so the published pages
show the personas the run used even after the records change.

Usage::

    pixi run publish-sample data-repository/results/<run>

Then run ``pixi run check``, which reads every published sample and fails on an unreadable
line, and commit the new folder.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

from corallab.paths import data_dir, resolve_data_path

logger = logging.getLogger(__name__)

ALLOWED = (
    re.compile(r"batch-[^/]+/[^/]+-sim\.jsonl"),
    re.compile(r"batch-[^/]+/[^/]+-check\.jsonl"),
    re.compile(r"batch-[^/]+/[^/]+-judge\.jsonl"),
    re.compile(r"batch-[^/]+/[^/]+\.trace\.jsonl"),
    re.compile(r"inputs/[^/]+\.(?:jsonl|yaml|yml|json)"),
    re.compile(r"inputs/persona_records/p-[a-z0-9-]+\.json"),
)
"""Paths, relative to a sample run's folder, that a published sample may hold."""


def allowed(relative: str) -> bool:
    return any(pattern.fullmatch(relative) for pattern in ALLOWED)


def samples_dir() -> Path:
    return data_dir() / "sample_results"


def _persona_ids(cases_file: Path) -> set[str]:
    ids: set[str] = set()
    for line in cases_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ref = str(json.loads(line).get("persona_ref") or "")
            if ref:
                ids.add(ref.partition("@")[0])
    return ids


def publish(run: Path, out_root: Path) -> Path:
    """Copy ``run`` into ``out_root/<run name>``, replacing an earlier copy."""
    out = out_root / run.name
    if out.exists():
        shutil.rmtree(out)
    for path in sorted(run.rglob("*")):
        relative = path.relative_to(run).as_posix()
        if path.is_file() and allowed(relative):
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    personas = Path(resolve_data_path("persona_records"))
    for cases_file in sorted((out / "inputs").glob("*.jsonl")):
        for persona_id in sorted(_persona_ids(cases_file)):
            source = personas / f"{persona_id}.json"
            if source.is_file():
                target = out / "inputs" / "persona_records" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            else:
                logger.warning("Persona record not found: %s", source)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path, help="A run folder under data-repository/results")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not (args.run / "inputs").is_dir():
        logger.error(
            "%s has no inputs/ folder; only a run with copied inputs can be published.", args.run
        )
        return 1
    out = publish(args.run, samples_dir())
    logger.info("Published %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
