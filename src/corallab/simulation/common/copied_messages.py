"""Detect a simulated person's message that copies a message from the AI it is talking to."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from functools import lru_cache

_SHARED = 0.9
"""A copy shares at least this fraction of the message it is measured against."""
_MIN_PART = 0.4
"""A part copied out of a longer message is at least this fraction of its length, or at least
``_LONG_PART`` characters, so a short phrase quoted back is not a copy."""
_LONG_PART = 80
_MIN_CHARS = 20
"""Messages shorter than this are never copies, so a shared "Okay, thanks." is not flagged."""


@lru_cache(maxsize=4096)
def _normalized(text: str) -> str:
    text = re.sub(r"[*_`\"“”‘’']", "", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def _shared(mine: str, theirs: str) -> int:
    matcher = SequenceMatcher(None, mine, theirs, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks())


def _copies(mine: str, theirs: str) -> bool:
    """Whether ``mine`` is ``theirs``, most of it, or all of it with a little added."""
    if len(mine) < _MIN_CHARS or len(theirs) < _MIN_CHARS:
        return False
    shared = _shared(mine, theirs)
    long_enough = len(mine) >= _MIN_PART * len(theirs) or len(mine) >= _LONG_PART
    part_of_theirs = shared >= _SHARED * len(mine) and long_enough
    all_of_theirs = shared >= _SHARED * len(theirs)
    return part_of_theirs or all_of_theirs


def copied_ai_message(message: str, history: Sequence[tuple[str, str]]) -> int | None:
    """Index into ``history`` of the AI message that ``message`` copies, or None.

    ``history`` is the conversation so far as ``(role, text)`` pairs with roles ``"user"``
    (the person) and ``"assistant"`` (the AI). A copy is a whole message, a part of one, or
    one with a little added. When ``message`` is mostly an earlier person message, an AI
    message that mirrors that earlier message is skipped, so a person who repeats their own
    words is not flagged. Markdown markers, quote characters, case and spacing are ignored.
    """
    mine = _normalized(message)
    if len(mine) < _MIN_CHARS:
        return None
    person: list[str] = []
    for i, (role, text) in enumerate(history):
        theirs = _normalized(text)
        if role == "user":
            person.append(theirs)
        elif role == "assistant":
            if any(
                _shared(mine, earlier) >= _SHARED * len(mine) and _copies(theirs, earlier)
                for earlier in person
                if earlier
            ):
                continue
            if _copies(mine, theirs):
                return i
    return None
