"""Which installed code produced a record written into the data repository."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PackageVersion(BaseModel):
    """One package that took part in producing a record."""

    name: str = Field(description="Distribution name, e.g. 'corallab'")
    version: str = Field(description="Installed package version")
    revision: str | None = Field(
        default=None,
        description="Git commit of the checkout, when this process is running from one",
    )
    dirty: bool | None = Field(
        default=None,
        description="True when that checkout has uncommitted changes; null when git could not say",
    )


class ProducedBy(BaseModel):
    """The toolkit and Knowledge Library that wrote a transcript or judge result."""

    toolkit: PackageVersion = Field(description="corallab build that wrote the record")
    knowledge_library: PackageVersion = Field(description="coralmap build loaded for the run")
