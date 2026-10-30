"""``corallab sim`` — generate conversation transcripts.

L1/L2: calls the system-under-test directly with scripted turns.
L3: runs persona-prompted simulation.

Processes test cases ``simulation.batch_size`` at a time. Each batch is written
to ``<run_dir>-<YYYYMMDD-HHMMSS>/batch-NNN/`` as a JSONL of Transcript records,
read by ``corallab judge`` together with the test case records. A relative
``run_dir`` is stored under the data repository, and each invocation appends a
timestamp so a later run does not overwrite an earlier one. Open the results
with ``pixi run viewer``.
"""

from __future__ import annotations

import asyncio
import json
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
from corallab.code_version import produced_by

if TYPE_CHECKING:
    from corallab.schemas.eval.run_case import RunCase
    from corallab.schemas.eval.run_manifest import RunManifest

logger = logging.getLogger(__name__)


def sim(
    ctx: typer.Context,
    run_manifest: Annotated[str, typer.Option(help="Path to run manifest YAML")],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Inspect SUT prompts and token counts without making API calls",
        ),
    ] = False,
) -> None:
    """Simulate conversations and generate transcripts for all test cases.

    When the manifest lists multiple ``system_under_test`` models, sim fans out
    over them: one transcript file per model (``<run_stem>__<key>-sim.jsonl``) plus
    a ``<run_stem>.index.json`` cataloging the set. Each output is directly
    consumable by ``corallab judge``.

    ``--dry-run`` short-circuits every model call (no API traffic), tallies the
    SUT prompt tokens, and prints a summary instead of writing transcripts.
    """
    from corallab.llm_client import reset_dry_run_context, set_dry_run_context
    from corallab.simulation.common.dry_run import DryRunCollector

    manifest = resolve_run_manifest(run_manifest)
    check_fidelity_prompt(manifest)
    out_dir, run_stem = init_logging(ctx, manifest)
    copy_run_inputs(run_manifest, manifest, out_dir)
    per_model = manifest.per_model_manifests()

    collector = DryRunCollector() if dry_run else None
    dry_run_token = set_dry_run_context(collector) if dry_run else None
    try:
        if len(per_model) == 1:
            paths, _, _run_cases = _execute_sim(per_model[0][1], out_dir, run_stem, dry_run=dry_run)
            if not dry_run:
                for out_path in paths:
                    console.print(f"Output: [cyan]{out_path}[/]")
            return

        console.print(f"[bold]Fanning out sim over {len(per_model)} models[/]")
        index_entries: list[dict] = []
        for key, sub in per_model:
            model_stem = f"{run_stem}__{key}"
            console.print(f"[bold]→ {sub.primary_sut.model_id}[/] ([cyan]{key}[/])")
            paths, _, _sub_cases = _execute_sim(sub, out_dir, model_stem, dry_run=dry_run)
            for out_path in paths:
                n_cases = 0
                if out_path.is_file():
                    n_cases = sum(1 for line in out_path.read_text().splitlines() if line.strip())
                index_entries.append(
                    {
                        "model_id": sub.primary_sut.model_id,
                        "output_key": key,
                        "transcript_path": str(out_path.relative_to(out_dir)),
                        "n_transcribed": n_cases,
                    }
                )

        if not dry_run:
            index_path = _write_index(out_dir, run_stem, manifest, index_entries)
            console.print(f"[bold green]Sim fan-out complete.[/] Index: [cyan]{index_path}[/]")
    finally:
        if dry_run_token is not None:
            reset_dry_run_context(dry_run_token)
        if dry_run and collector is not None:
            console.print(collector.render_summary())


def _execute_sim(
    manifest: RunManifest,
    out_dir: Path,
    run_stem: str,
    dry_run: bool = False,
) -> tuple[list[Path], int, list[RunCase]]:
    """Run the sim batch graph via Burr.

    Returns (batch output paths, n transcribed, run cases). Each batch file
    lives in ``out_dir/batch-NNN/``.
    """
    from corallab.cli._runners import run_batch_sim

    out_path = out_dir / f"{run_stem}-sim.jsonl"
    resolved_db = out_dir / f"{run_stem}-sim.burr.db"

    try:
        result = asyncio.run(
            run_batch_sim(
                manifest,
                manifest.test_case_records_path,
                out_path,
                resolved_db,
                run_stem,
                dry_run=dry_run,
            )
        )
    except Exception as exc:
        logger.error("Fatal error in sim batch: %s", exc, exc_info=True)
        raise typer.Exit(1) from exc

    sim_outputs, run_cases, accumulator, paths = result

    if dry_run:
        return paths, len(sim_outputs), run_cases

    if not sim_outputs:
        err_console.print("[bold red]Error:[/] no test cases produced transcripts")
        raise typer.Exit(1)
    console.print(
        f"[bold green]Simulation complete.[/] [cyan]{len(sim_outputs)}[/] test case(s) "
        f"in [cyan]{len(paths)}[/] batch(es)"
    )
    print_token_summary(accumulator, "sim")
    return paths, len(sim_outputs), run_cases


def _write_index(
    out_dir: Path,
    run_stem: str,
    manifest: RunManifest,
    entries: list[dict],
) -> Path:
    """Write the per-model transcript index for a multi-model sim fan-out."""
    index_path = out_dir / f"{run_stem}.index.json"
    index_path.write_text(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "evaluation_level": manifest.evaluation_level,
                "input_path": manifest.test_case_records_path,
                "produced_by": produced_by().model_dump(),
                "models": entries,
            },
            indent=2,
        )
        + "\n"
    )
    return index_path
