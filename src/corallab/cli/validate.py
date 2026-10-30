"""``corallab validate`` — check authored YAML and JSON with the Pydantic models.

``corallab validate test-cases`` checks a file of test case records.
``corallab validate run-manifest`` checks a manifest.
``corallab validate test-case-consistency`` reads every part of each test case against its
expected behaviors with a model: one paid call per test case.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from coralmap import LibraryParseError
from pydantic import ValidationError
from rich.markup import escape

from corallab.cli._helpers import console, err_console, read_run_manifest

logger = logging.getLogger(__name__)

validate_app = typer.Typer(
    name="validate",
    help=(
        "Validate authored YAML / JSON files against the Pydantic models in src/corallab/schemas/."
    ),
    no_args_is_help=True,
)


@validate_app.command("test-cases")
def validate_test_cases(
    path: Annotated[Path, typer.Argument(help="Path to a JSONL file of test case records")],
    personas: Annotated[
        str, typer.Option(help="Directory of persona records the test case records name")
    ] = "persona_records",
) -> None:
    """Validate test case records: models, personas, library pin, and expected behaviors."""
    from corallab.kl import load_library
    from corallab.simulation.data.test_case_records import (
        TestCaseRecordsError,
        check_test_case_records,
        load_test_case_records,
    )

    try:
        library = load_library()
        cases = load_test_case_records(path, personas, library)
    except (LibraryParseError, TestCaseRecordsError, FileNotFoundError) as exc:
        err_console.print(f"[bold red]Invalid:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    problems = check_test_case_records(cases, library)
    if problems:
        err_console.print(f"[bold red]Invalid:[/] {escape(str(path))}: {len(problems)} problem(s)")
        for problem in problems:
            err_console.print(f"  {escape(problem)}")
        raise typer.Exit(1)
    console.print(f"[green]OK[/] {escape(str(path))}: {len(cases)} test case(s)")


@validate_app.command("run-manifest")
def validate_run_manifest(
    path: Annotated[Path, typer.Argument(help="Path to a run manifest YAML file")],
) -> None:
    """Validate a run manifest YAML file against RunManifest."""
    from corallab.schemas.eval.run_manifest import RunManifest

    try:
        RunManifest.model_validate(read_run_manifest(path))
    except (ValidationError, ValueError, OSError) as exc:
        err_console.print(f"[bold red]Invalid:[/] {escape(str(path))}\n{escape(str(exc))}")
        raise typer.Exit(1) from exc
    console.print(f"[green]OK[/] {escape(str(path))} (validated by RunManifest)")


@validate_app.command("test-case-consistency")
def validate_test_case_consistency(
    path: Annotated[Path, typer.Argument(help="Path to a JSONL file of test case records")],
    model: Annotated[str, typer.Option(help="Model that reads the test cases (one call per case)")],
    temperature: Annotated[
        float | None, typer.Option(help="Sampling temperature; omitted when not set")
    ] = None,
    reasoning_effort: Annotated[
        str | None, typer.Option(help="Reasoning level for reasoning models")
    ] = None,
    personas: Annotated[
        str, typer.Option(help="Directory of persona records the test case records name")
    ] = "persona_records",
    out: Annotated[
        Path | None, typer.Option(help="Write every finding as JSONL to this file")
    ] = None,
) -> None:
    """Flag parts of each test case that work against its expected behaviors.

    Reads each test case's strategies, assertions, opening, scenario and persona against the
    expected behaviors' catalog definitions. Exits 1 when a part conflicts with an expected
    behavior or the check failed for a case. A part at risk is printed but does not fail.
    """
    from corallab.authoring.consistency_check import check_cases
    from corallab.kl import load_library
    from corallab.simulation.data.test_case_records import (
        TestCaseRecordsError,
        load_test_case_records,
    )

    try:
        library = load_library()
        cases = load_test_case_records(path, personas, library)
    except (LibraryParseError, TestCaseRecordsError, FileNotFoundError) as exc:
        err_console.print(f"[bold red]Invalid:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else None
    results = asyncio.run(
        check_cases(
            cases,
            library,
            model=model,
            temperature=temperature,
            extra_params=extra,
        )
    )
    if out is not None:
        out.write_text("".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8")
    failed = False
    for result in results:
        name = escape(f"{result.test_case_id} {result.display_name}".strip())
        if result.error:
            failed = True
            err_console.print(f"[bold red]Check failed:[/] {name}: {escape(result.error)}")
            continue
        flagged = result.conflicts + result.risks
        if not flagged:
            console.print(f"[green]OK[/] {name}")
            continue
        failed = failed or bool(result.conflicts)
        console.print(f"[bold]{name}[/]")
        for finding in flagged:
            colour = "red" if finding.verdict == "conflicts" else "yellow"
            quote = f' "{escape(finding.quote)}"' if finding.quote else ""
            unverified = (
                "" if finding.quote_found in (None, True) else " (quote not found in the entry)"
            )
            part = f"strategy {finding.number}" if finding.part == "strategy" else finding.part
            console.print(
                f"  [{colour}]{finding.verdict}[/] {part}"
                f" · {finding.behavior or 'no behavior named'}{quote}{unverified}"
            )
            console.print(f"    {escape(finding.reason)}")
            if finding.suggestion:
                console.print(f"    [dim]suggestion:[/] {escape(finding.suggestion)}")
    if out is not None:
        console.print(f"Findings written to [cyan]{escape(str(out))}[/]")
    if failed:
        raise typer.Exit(1)
