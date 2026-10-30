"""How the toolkit loads the Knowledge Library.

This module is the only place in the toolkit that calls ``coralmap.load``,
so there is exactly one answer to "which library is this run reading?". The
load is strict: it returns the whole library or raises
``coralmap.LibraryParseError`` naming every entry that failed to parse.
"""

from __future__ import annotations

from pathlib import Path

from coralmap import KnowledgeLibrary, load


def load_library(path: str | Path | None = None) -> KnowledgeLibrary:
    """Load the Knowledge Library at ``path``, or the one ``coralmap.load`` resolves.

    ``path`` resolves as ``coralmap.load`` resolves it: the argument, then
    ``CORALMIND_KL_PATH``, then the entries bundled in the installed package.
    """
    return load(path)
