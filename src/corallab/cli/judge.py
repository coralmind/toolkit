"""``corallab judge`` — evaluate existing transcripts with parallel judges.

Processes every transcript via the judge batch graph
(LoadJudgeInputs → ParallelJudge → WriteResults). Input is the Transcript JSONL from
``corallab sim``, and the manifest's test case records (``test_case_records_path``) and persona
directory (``persona_records_path``), which the transcripts are joined to.

After writing results, the JSONL is ready for the results viewer
(``pixi run viewer``).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from corallab.cli._helpers import (
    console,
    err_console,
    init_logging,
    print_token_summary,
    resolve_run_manifest,
    transcript_run_dir,
)

if TYPE_CHECKING:
    from corallab.schemas.eval.run_manifest import RunManifest

logger = logging.getLogger(__name__)


def judge(
    ctx: typer.Context,
    run_manifest: Annotated[str, typer.Option(help="Path to run manifest YAML")],
    transcripts: Annotated[
        str, typer.Option(help="Path to the Transcript JSONL written by `corallab sim`")
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-n",
            help="Inspect prompts and token counts without making API calls",
        ),
    ] = False,
) -> None:
    """Evaluate existing transcripts by running judges in parallel.

    Writes a judge-results JSONL. Open it in the results viewer.
    """
    manifest = resolve_run_manifest(run_manifest)
    # Judge the transcripts where they already live, instead of opening a second
    # timestamped folder that the viewer would treat as a different run. Results go
    # beside the transcript file, so judging one batch never replaces another's.
    init_logging(ctx, manifest, out_dir=transcript_run_dir(transcripts))
    source = Path(transcripts)
    out_path = _execute_judge(
        manifest, transcripts, source.parent, source.stem.removesuffix("-sim"), dry_run=dry_run
    )
    if not dry_run:
        console.print(f"[bold green]Done.[/] Results written to [cyan]{out_path}[/]")


def _execute_judge(
    manifest: RunManifest,
    transcripts: str,
    out_dir: Path,
    run_stem: str,
    dry_run: bool = False,
) -> Path:
    """Run the judge batch graph via Burr. Returns the output path."""
    from corallab.cli._runners import run_batch_judge
    from corallab.llm_client import reset_dry_run_context, set_dry_run_context
    from corallab.simulation.common.dry_run import DryRunCollector

    out_path = out_dir / f"{run_stem}-judge.jsonl"
    resolved_db = out_dir / f"{run_stem}-judge.burr.db"

    collector = DryRunCollector() if dry_run else None
    dry_run_token = set_dry_run_context(collector) if dry_run else None

    try:
        results, accumulator = asyncio.run(
            run_batch_judge(
                manifest,
                transcripts,
                out_path,
                resolved_db,
                f"{run_stem}-judge",
                dry_run=dry_run,
            )
        )
    except Exception as exc:
        logger.error("Fatal error in judge batch: %s", exc, exc_info=True)
        raise typer.Exit(1) from exc
    finally:
        if dry_run_token is not None:
            reset_dry_run_context(dry_run_token)
        if dry_run and collector is not None:
            # Print even on partial failure — collector has all calls up to the error point.
            console.print(collector.render_summary())

    if dry_run:
        return out_path

    if not results:
        err_console.print("[bold red]Error:[/] no results produced")
        raise typer.Exit(1)

    console.print(f"[bold green]Judge complete.[/] [cyan]{len(results)}[/] result(s) written")
    print_token_summary(accumulator, "judge")
    return out_path
