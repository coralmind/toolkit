"""Produced records name the toolkit and Knowledge Library that wrote them."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

import coralmap

from corallab.code_version import produced_by


def test_produced_by_names_the_installed_packages() -> None:
    stamp = produced_by()
    assert stamp.toolkit.name == "corallab"
    assert stamp.toolkit.version == importlib.metadata.version("corallab")
    assert stamp.knowledge_library.name == "coralmap"
    assert stamp.knowledge_library.version == coralmap.__version__
    assert stamp.toolkit.revision is not None
    assert re.fullmatch(r"[0-9a-f]{40}", stamp.toolkit.revision)
    # coralmap installs from PyPI. A wheel has no checkout, so the revision
    # stays null. An editable install of the library records that commit.
    library = Path(coralmap.__file__).resolve()
    if "site-packages" in library.parts or "dist-packages" in library.parts:
        assert stamp.knowledge_library.revision is None
        assert stamp.knowledge_library.dirty is None
    else:
        assert stamp.knowledge_library.revision is not None
        assert re.fullmatch(r"[0-9a-f]{40}", stamp.knowledge_library.revision)
