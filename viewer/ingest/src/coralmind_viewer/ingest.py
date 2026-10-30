"""Read a results folder into the generated page data, once or whenever it changes.

``ingest`` writes once and exits non-zero naming the file it could not read. ``watch``
re-ingests when the results change, after writes have been quiet for a second; a failed
re-ingest keeps the last good data and records why in ``status.json``. ``dev`` runs the
watcher beside ``npm run dev``.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from coralmind_viewer import export_models as M
from coralmind_viewer import facts as F
from coralmind_viewer.export import Writer, export

logger = logging.getLogger(__name__)

POLL_S = 0.5
QUIET_S = 1.0


def default_results() -> Path:
    from corallab.paths import results_dir

    return results_dir()


def default_out() -> Path:
    from corallab.paths import repo_root

    return repo_root() / "viewer" / "src" / "generated"


class IngestError(Exception):
    """The results could not be read or written; the message names the file."""


def _write_status(out: Path, status: M.StatusData) -> None:
    Writer(out).write("status.json", status)


def ingest_once(index: F.Index, out: Path) -> Writer:
    """Rebuild ``index`` and export it. Raises :class:`IngestError` and keeps the old data."""
    snapshot = index.rebuild()
    state = index.status()
    error = state.first_error or state.stale_error
    if error or snapshot is None:
        raise IngestError(error or "no snapshot")
    try:
        return export(snapshot, out)
    except (OSError, F.StaleRecord) as exc:
        raise IngestError(f"{type(exc).__name__}: {exc}") from exc


def run_ingest(results: Path, out: Path) -> int:
    index = F.Index(results)
    try:
        writer = ingest_once(index, out)
    except IngestError as exc:
        logger.error("Could not read results: %s", exc)
        return 1
    _write_status(out, M.StatusData(stale_since=None, error=None))
    logger.info("Ingested %s into %s (%d files changed)", results, out, writer.changed)
    return 0


def watch(
    results: Path,
    out: Path,
    *,
    on_ready: Callable[[], None] | None = None,
    stop: float | None = None,
) -> None:
    """Ingest, call ``on_ready``, then re-ingest after each quiet change.

    Runs until interrupted, or until ``stop`` on the monotonic clock.
    """
    index = F.Index(results)
    stale_since: str | None = None

    def once() -> None:
        nonlocal stale_since
        try:
            writer = ingest_once(index, out)
        except IngestError as exc:
            stale_since = stale_since or datetime.now().isoformat(timespec="seconds")
            logger.warning("Re-ingest failed; keeping the last good data: %s", exc)
            _write_status(out, M.StatusData(stale_since=stale_since, error=str(exc)))
            return
        stale_since = None
        _write_status(out, M.StatusData(stale_since=None, error=None))
        if writer.changed:
            logger.info("Results changed: %d files rewritten", writer.changed)

    results.mkdir(parents=True, exist_ok=True)
    once()
    if on_ready is not None:
        on_ready()
    seen = F._signature(results)
    changed_at: float | None = None
    while stop is None or time.monotonic() < stop:
        time.sleep(POLL_S)
        now = F._signature(results)
        if now != seen:
            seen, changed_at = now, time.monotonic()
        elif changed_at is not None and time.monotonic() - changed_at >= QUIET_S:
            changed_at = None
            once()


def run_dev(results: Path, out: Path, viewer_dir: Path) -> int:
    """``npm run dev`` in ``viewer_dir`` with the watcher beside it, started after the ingest."""
    servers: list[subprocess.Popen[bytes]] = []

    def start() -> None:
        servers.append(subprocess.Popen(["npm", "run", "dev"], cwd=viewer_dir))

    try:
        watch(results, out, on_ready=start)
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
    return 0
