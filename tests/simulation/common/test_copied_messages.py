"""A person message that copies the AI is found; quotes, short replies and mirroring are not."""

from __future__ import annotations

import pytest

from corallab.simulation.common.copied_messages import copied_ai_message

_AI = (
    "You're welcome. Being “not keen on it” is completely understandable, but the fact "
    "that you feel a little better is a good sign.\n\n"
    "You don't have to rush, and you don't have to feel enthusiastic about it. Just aim for "
    "**small, ordinary steps** and let “good enough” be enough.\n\n"
    "If you want, I can help you make a very simple plan for the next time you use the break "
    "room so it feels less daunting."
)
_PERSON = "Thank you. I do feel a little better hearing it put that way."
_HISTORY = [("user", _PERSON), ("assistant", _AI)]
_PARAGRAPHS = _AI.split("\n\n")


@pytest.mark.parametrize(
    "message",
    [
        _AI,
        _AI.replace("**", "").replace("“", '"').replace("”", '"').upper(),
        "\n\n".join(_PARAGRAPHS[:2]),
        _PARAGRAPHS[0],
        f"{_AI} Thanks, I think.",
    ],
    ids=[
        "whole",
        "other-markdown-and-case",
        "minus-last-paragraph",
        "one-paragraph",
        "plus-a-line",
    ],
)
def test_a_copy_of_the_ai_names_the_copied_message(message: str) -> None:
    assert copied_ai_message(message, _HISTORY) == 1


@pytest.mark.parametrize(
    "message",
    [
        "Good enough is enough, I suppose.",
        "Small, ordinary steps. I can try that, I suppose. Thanks for being patient.",
        "I don't have to rush, you're right. Just aim for small, ordinary steps. "
        "Maybe I'll sit with Carol on Monday.",
        _PERSON,
        "Okay, thanks.",
    ],
    ids=["short-quote", "own-words-with-a-phrase", "long-own-words", "own-earlier-line", "short"],
)
def test_the_persons_own_words_are_not_a_copy(message: str) -> None:
    assert copied_ai_message(message, _HISTORY) is None


def test_a_short_ai_reply_is_never_copied() -> None:
    assert (
        copied_ai_message("Okay, thanks. That helps a lot.", [("assistant", "Okay, thanks.")])
        is None
    )


def test_repeating_yourself_after_the_ai_mirrored_you_is_not_a_copy() -> None:
    history = [
        ("user", "I just can't stop thinking about what he said to me last night."),
        ("assistant", "You just can't stop thinking about what he said to you last night."),
    ]
    repeat = "I just cannot stop thinking about what he said to me last night."
    assert copied_ai_message(repeat, history) is None


def test_copying_an_ai_reply_that_quotes_you_is_a_copy() -> None:
    said = "I just cannot stop thinking about the accident at night."
    reply = (
        f'You said "{said}" That sounds exhausting, and it makes sense that nights are '
        "the hardest part. Would it help to talk through what usually happens before sleep?"
    )
    assert copied_ai_message(reply, [("user", said), ("assistant", reply)]) == 1
    assert copied_ai_message(said, [("user", said), ("assistant", reply)]) is None


def test_the_index_points_at_the_copied_reply_among_several() -> None:
    history = [*_HISTORY, ("user", "Okay."), ("assistant", "Would a plan for Monday help?")]
    assert copied_ai_message(_AI, history) == 1
