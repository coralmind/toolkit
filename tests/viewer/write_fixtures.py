"""Write the vitest fixtures: every scenario in ``scenarios.py``, read by the ingest.

Each scenario gets ``viewer/tests/fixtures/generated/<name>/`` holding ``data/`` (its data
repository and results) and ``gen/`` (what the ingest wrote), plus ``meta.json`` naming its
suites and transcripts. Run it with ``pixi run -e viewer viewer-fixtures``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from scenarios import KL_FIXTURE, SCENARIOS, results  # noqa: E402

from coralmind_viewer import export as E  # noqa: E402
from coralmind_viewer import facts as F  # noqa: E402

OUT = HERE.parents[1] / "viewer" / "tests" / "fixtures" / "generated"


def write(name: str) -> None:
    item = SCENARIOS[name]
    base = OUT / name
    shutil.rmtree(base, ignore_errors=True)
    data = base / "data"
    data.mkdir(parents=True)
    os.environ["CORALMIND_DATA_REPOSITORY_PATH"] = str(data)
    os.environ["CORALMIND_KL_PATH"] = str(KL_FIXTURE)
    kl = item.kl_path(data) if item.kl_path is not None else None
    if kl is not None and item.records_use_kl_path:
        os.environ["CORALMIND_KL_PATH"] = str(kl)
    item.build(data)
    if kl is not None:
        os.environ["CORALMIND_KL_PATH"] = str(kl)
    index = F.Index(results(data))
    snapshot = index.rebuild()
    assert snapshot is not None, index.status().first_error
    E.export(snapshot, base / "gen")
    if item.after is not None:
        os.environ["CORALMIND_KL_PATH"] = str(KL_FIXTURE)
        item.after(data)
    meta = {
        "suites": sorted(snapshot.suites),
        "names": sorted(snapshot.suite_files),
        "transcripts": [{"run": s.run, "id": s.transcript_id} for s in snapshot.sims],
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    for name in SCENARIOS:
        write(name)
    print(f"Wrote {len(SCENARIOS)} scenarios to {OUT}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
