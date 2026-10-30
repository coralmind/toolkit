"""Message schemas — conversation turns."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# TODO: type using litellm types
# TODO: Check with Jina if we need all properties.
class Turn(BaseModel):
    """One message in a conversation (used by both pipeline and simulation).

    A turn is one exchange: the user message and the assistant reply that answers
    it share ``turn_number``. The opening user message is turn 1.
    """

    turn_number: int = Field(
        ge=1,
        description=(
            "1-indexed exchange. The user message and the assistant reply that answers "
            "it share this number. The opening user message is 1"
        ),
    )
    role: Literal["user", "assistant"] = Field(
        description="Speaker of this message. Both messages of an exchange share turn_number"
    )
    content: str = Field(description="Message text content")
    timestamp: str | None = Field(default=None, description="ISO 8601 timestamp of this turn")
