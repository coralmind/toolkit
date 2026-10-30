"""corallab CLI — entry point for the AI Mental Health Evaluation Pipeline.

Subcommands:
    corallab sim          — Simulate conversations, write Transcript records
    corallab check        — Run the fidelity check on existing transcripts
    corallab judge        — Judge existing transcripts joined to the test case records
    corallab run          — sim + check + judge in one command
    corallab check-env    — Confirm .env loading + report which keys are set
    corallab run-preview  — Visualize the evaluation Burr graph
    corallab validate     — Validate test case records or a run manifest
"""

from __future__ import annotations

from typing import Annotated

import typer
from dotenv import load_dotenv

from corallab.cli._helpers import LogLevel

app = typer.Typer(
    name="corallab",
    help="AI Mental Health Evaluation Pipeline",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def _callback(
    ctx: typer.Context,
    log_level: Annotated[
        LogLevel, typer.Option("--log-level", help="Logging level")
    ] = LogLevel.INFO,
) -> None:
    """AI Mental Health Evaluation Pipeline."""
    load_dotenv()
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level


def _register_commands() -> None:
    from corallab.cli.check import check
    from corallab.cli.check_env import check_env
    from corallab.cli.judge import judge
    from corallab.cli.run import run
    from corallab.cli.run_preview import run_preview
    from corallab.cli.sim import sim
    from corallab.cli.validate import validate_app

    app.command()(sim)
    app.command()(check)
    app.command()(judge)
    app.command()(run)
    app.command("check-env")(check_env)
    app.command("run-preview")(run_preview)
    app.add_typer(validate_app, name="validate")


_register_commands()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
