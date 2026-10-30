"""Knowledge Library pin — which library a test case or run was read from."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from coralmap import KnowledgeLibrary


class KlPin(BaseModel):
    """Which Knowledge Library a run, result, or test case was produced against."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(description="coralmap package version (the release id)")
    digest: str = Field(description="Content hash of the loaded entries")

    @classmethod
    def from_library(cls, library: KnowledgeLibrary) -> KlPin:
        """Pin the library a run actually read."""
        return cls(version=library.version, digest=library.digest)
