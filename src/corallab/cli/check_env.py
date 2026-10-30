"""``corallab check-env`` — confirm ``.env`` loads and which keys are set.

Loads ``.env`` the way the CLI callback does, then reports presence (never
values) of every env var the LLM endpoint registry references.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated

import typer
import yaml
from dotenv import find_dotenv
from rich.table import Table

from corallab import paths
from corallab.cli._helpers import console

_ENV_TOKEN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def referenced_vars(registry_path: Path) -> list[str]:
    """Sorted env-var names the registry references.

    Scrapes ``api_key_env`` names and ``${VAR}`` interpolation tokens from each
    endpoint's ``api_base`` (raw YAML — never the resolved secret values).
    A missing registry references nothing.
    """
    names: set[str] = set()
    if registry_path.is_file():
        raw = yaml.safe_load(registry_path.read_text()) or {}
        for cfg in (raw.get("endpoints") or {}).values():
            if not isinstance(cfg, dict):
                continue
            if isinstance(cfg.get("api_key_env"), str):
                names.add(cfg["api_key_env"])
            names.update(_ENV_TOKEN.findall(cfg.get("api_base") or ""))
    return sorted(names)


def check_env(
    registry: Annotated[
        Path | None,
        typer.Option("--registry", help="Endpoint registry to scan (default: endpoints.yaml)."),
    ] = None,
) -> None:
    """Confirm ``.env`` is loaded and report which referenced env vars are set.

    The top-level CLI callback already called ``load_dotenv()``, so the
    environment is populated before this runs — we only locate the file (for
    the report) and read presence. Values are never printed.
    """
    dotenv_path = find_dotenv()
    console.print(f".env resolved : [cyan]{dotenv_path or '(none found)'}[/cyan]\n")

    registry_path = registry or paths.endpoints_registry_path()
    names = referenced_vars(registry_path)

    table = Table(title="Referenced env vars (values never shown)")
    table.add_column("status", justify="center")
    table.add_column("variable", style="cyan")
    n_set = 0
    for var in names:
        is_set = bool(os.environ.get(var))
        n_set += int(is_set)
        table.add_row("[green]SET[/green]" if is_set else "[dim]unset[/dim]", var)
    console.print(table)
    console.print(f"\n[bold]{n_set}/{len(names)}[/bold] set.")

    if not dotenv_path:
        raise typer.Exit(code=1)
