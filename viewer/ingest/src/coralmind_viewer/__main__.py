"""``python -m coralmind_viewer`` — read results into the Next app's generated page data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from coralmind_viewer import ingest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest results for the CoralMind results viewer")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("ingest", "Read the results once into the generated page data"),
        ("watch", "Re-ingest whenever the results change"),
        ("dev", "Run the watcher beside the Next dev server"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument(
            "--results",
            type=Path,
            default=None,
            help="Results directory to read (default: the data repository's results/)",
        )
        command.add_argument("--out", type=Path, default=None, help="Generated data directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = args.results or ingest.default_results()
    out = args.out or ingest.default_out()
    if args.command == "ingest":
        sys.exit(ingest.run_ingest(results, out))
    if args.command == "watch":
        ingest.watch(results, out)
        return
    sys.exit(ingest.run_dev(results, out, out.parents[1]))


if __name__ == "__main__":
    main()
