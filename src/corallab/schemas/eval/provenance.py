"""Identifiers shared by the persona, test case, and transcript records.

A persona's or test case's ``provenance`` is where its producer says how it was made.
The models interpret only its ``kind``; anything else a producer records there is
checked by that producer's own tooling.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

KL_ID = r"^kl-\d{6}$"
PERSONA_ID = r"^p-[a-z0-9]+(?:-[a-z0-9]+)*$"
PERSONA_REF = r"^p-[a-z0-9]+(?:-[a-z0-9]+)*@[1-9]\d*$"
TEST_CASE_ID = r"^aimhef-[a-f0-9]{8}$"

FactMode = Literal["stated", "asserted", "withheld"]


class Provenance(BaseModel):
    """How a record was made, as its producer records it.

    Only ``kind`` has a fixed meaning. Any other key is the producer's own record (a
    design row, the model call that wrote the text, a name replacement) and is checked
    by the producer's tooling, not by these models.
    """

    model_config = ConfigDict(extra="allow")

    kind: Literal["authored", "generated", "imported"] | None = Field(
        default=None,
        description=(
            "'authored' by a person, 'generated' by a program, or 'imported' from another dataset"
        ),
    )
