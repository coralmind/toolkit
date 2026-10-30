"""``corallab run`` — sim, then check, then judge, in a single command.

Runs three sequential stages:
1. **sim** — generate conversation transcripts (Transcript records)
2. **check** — the fidelity check on each batch (L3, when ``elicitation_fidelity.enabled``)
3. **judge** — evaluate transcripts with parallel judges (EvalResults)

Read the results with the viewer (``pixi run viewer``).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from corallab.cli._helpers import (
    check_fidelity_prompt,
    console,
    copy_run_inputs,
    err_console,
    init_logging,
    print_token_summary,
    resolve_run_manifest,
)

if TYPE_CHECKING:
    from corallab.schemas.eval.run_manifest import RunManifest

logger = logging.getLogger(__name__)


def run(
    ctx: typer.Context,
    run_manifest: Annotated[str, typer.Option(help="Path to run manifest YAML")],
    db_path: Annotated[
        str | None, typer.Option(help="Path to SQLite DB for state persistence")
    ] = None,
) -> None:
    """Run the full pipeline: simulate, then judge.

    When the manifest lists multiple ``system_under_test`` models, the whole
    pipeline fans out over them — each model gets its own sim → judge with
    per-model output stems (``<run_stem>__<key>…``).
    """
    manifest = resolve_run_manifest(run_manifest)
    check_fidelity_prompt(manifest)
    out_dir, run_stem = init_logging(ctx, manifest)
    copy_run_inputs(run_manifest, manifest, out_dir)
    per_model = manifest.per_model_manifests()

    if len(per_model) == 1:
        _run_pipeline(per_model[0][1], out_dir, run_stem, db_path)
        return

    console.print(f"[bold]Fanning out full pipeline over {len(per_model)} models[/]")
    for i, (key, sub) in enumerate(per_model, start=1):
        console.print(
            f"[bold]Model {i}/{len(per_model)}: {sub.primary_sut.model_id}[/] ([cyan]{key}[/])"
        )
        # Per-model state DBs avoid collisions; ignore any shared --db-path here.
        _run_pipeline(sub, out_dir, f"{run_stem}__{key}", None)


def _run_pipeline(
    manifest: RunManifest,
    out_dir: Path,
    run_stem: str,
    db_path: str | None,
) -> None:
    """Run sim → check → judge for a single (single-model) manifest."""
    from corallab.cli.check import _execute_check

    # Stage 1: sim. Groups are already on disk, each in its own directory.
    console.print("[bold]Stage 1/3:[/] Simulation")
    sim_out = _execute_sim(manifest, out_dir, run_stem)
    sim_paths = list(sim_out)

    for index, sim_out_path in enumerate(sim_paths, start=1):
        group_dir = sim_out_path.parent
        if len(sim_paths) > 1:
            console.print(f"[bold]Batch {index}/{len(sim_paths)}[/] [cyan]{group_dir.name}[/]")
        # Stage 2: check this group's transcripts, when the manifest asks for it.
        if manifest.evaluation_level >= 3 and manifest.simulation.elicitation_fidelity.enabled:
            console.print("[bold]Stage 2/3:[/] Fidelity check")
            _execute_check(manifest, sim_out_path, group_dir, run_stem)
        # Stage 3: judge this group's transcripts. Judge's own in-flight cap is unchanged.
        console.print("[bold]Stage 3/3:[/] Judge")
        # One shared --db-path cannot hold every group. Each group keeps its own file.
        judge_db = db_path if len(sim_paths) == 1 else None
        _execute_judge(manifest, sim_out_path, judge_db, group_dir, run_stem)


def _execute_sim(
    manifest: RunManifest,
    out_dir: Path,
    run_stem: str,
) -> list[Path]:
    """Run the sim batch graph. Returns one transcript JSONL path per batch."""
    from corallab.cli._runners import run_batch_sim

    sim_stem = f"{run_stem}-sim"
    out_path = out_dir / f"{sim_stem}.jsonl"
    resolved_db = out_dir / f"{sim_stem}.burr.db"

    try:
        result = asyncio.run(
            run_batch_sim(
                manifest, manifest.test_case_records_path, out_path, resolved_db, sim_stem
            )
        )
    except Exception as exc:
        logger.error("Fatal error in sim stage: %s", exc, exc_info=True)
        err_console.print(f"[bold red]Sim failed:[/] {exc}")
        raise typer.Exit(1) from exc

    sim_outputs, _, accumulator, paths = result

    if not sim_outputs:
        err_console.print("[bold red]Error:[/] no test cases produced transcripts")
        raise typer.Exit(1)
    console.print(
        f"[bold green]Sim complete.[/] "
        f"[cyan]{len(sim_outputs)}[/] test case(s) in [cyan]{len(paths)}[/] batch(es)"
    )
    for path in paths:
        console.print(f"  [cyan]{path}[/]")
    print_token_summary(accumulator, "sim")
    return paths


def _execute_judge(
    manifest: RunManifest,
    sim_out_path: Path,
    db_path: str | None,
    out_dir: Path,
    run_stem: str,
) -> Path:
    """Run the judge batch graph on sim output. Returns the path to the results JSONL."""
    from corallab.cli._runners import run_batch_judge

    judge_stem = f"{run_stem}-judge"
    out_path = out_dir / f"{judge_stem}.jsonl"
    resolved_db = Path(db_path) if db_path else out_dir / f"{judge_stem}.burr.db"

    try:
        results, accumulator = asyncio.run(
            run_batch_judge(manifest, str(sim_out_path), out_path, resolved_db, judge_stem)
        )
    except Exception as exc:
        logger.error("Fatal error in judge stage: %s", exc, exc_info=True)
        err_console.print(f"[bold red]Judge failed:[/] {exc}")
        raise typer.Exit(1) from exc

    if not results:
        err_console.print("[bold red]Error:[/] no results produced")
        raise typer.Exit(1)

    logger.info("Wrote %d results to %s", len(results), out_path)
    console.print(
        f"[bold green]Judge complete.[/] [cyan]{len(results)}[/] result(s) → [cyan]{out_path}[/]"
    )
    print_token_summary(accumulator, "judge")
    return out_path
