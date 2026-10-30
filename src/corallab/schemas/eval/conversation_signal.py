"""ConversationSignal — structured output from PersonResponseGenerator."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["ConversationSignal"]


class ConversationSignal(BaseModel):
    conversation_closing: bool = False
    scenario_enacted: bool = False
