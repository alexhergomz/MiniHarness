"""Reasoning-block handling.

Local reasoning models (Qwen3, R1 distills) emit chain-of-thought inline in
``content``. Observed live via ollama/qwen: without filtering it lands in the
conversation history, where it costs context on every later turn and feeds the
model its own half-formed reasoning.

The hard part is that tags straddle streaming chunk boundaries.
"""

from __future__ import annotations

import pytest

from miniharness.provider import AssistantTurn, ThinkFilter


def feed_all(chunks):
    """Run chunks through a filter, return (visible, thinking)."""
    f = ThinkFilter()
    vis, think = [], []
    for c in chunks:
        v, t = f.feed(c)
        vis.append(v)
        think.append(t)
    v, t = f.flush()
    vis.append(v)
    think.append(t)
    return "".join(vis), "".join(think)


def test_plain_text_passes_through_untouched():
    assert feed_all(["hello ", "world"]) == ("hello world", "")


def test_a_whole_think_block_in_one_chunk():
    vis, think = feed_all(["<think>reasoning</think>answer"])
    assert vis == "answer"
    assert think == "reasoning"


def test_text_before_and_after_a_think_block():
    vis, think = feed_all(["before <think>hmm</think> after"])
    assert vis == "before  after"
    assert think == "hmm"


@pytest.mark.parametrize("split", range(1, 7))
def test_open_tag_split_across_chunks(split):
    """`<think>` arriving as `<th` + `ink>` must not leak into visible text."""
    s = "<think>"
    vis, think = feed_all([s[:split], s[split:], "secret", "</think>", "shown"])
    assert vis == "shown", f"leaked at split {split}"
    assert think == "secret"


@pytest.mark.parametrize("split", range(1, 8))
def test_close_tag_split_across_chunks(split):
    s = "</think>"
    vis, think = feed_all(["<think>", "secret", s[:split], s[split:], "shown"])
    assert vis == "shown"
    assert think == "secret"


def test_one_character_at_a_time():
    """The pathological case: every tag character in its own chunk."""
    text = "a<think>b</think>c"
    vis, think = feed_all(list(text))
    assert vis == "ac"
    assert think == "b"


def test_multiple_think_blocks():
    vis, think = feed_all(["<think>one</think>A<think>two</think>B"])
    assert vis == "AB"
    assert think == "onetwo"


def test_unterminated_think_block_is_not_leaked_as_text():
    """A truncated response can end mid-thought; that must not become output."""
    vis, think = feed_all(["<think>still reasoning when the tokens ran out"])
    assert vis == ""
    assert "still reasoning" in think


def test_a_dangling_partial_tag_is_flushed_as_text():
    """`<thi` at end of stream was never a tag, so it is real output."""
    vis, think = feed_all(["result <thi"])
    assert vis == "result <thi"
    assert think == ""


def test_angle_brackets_that_are_not_think_tags_survive():
    vis, think = feed_all(["use Vec<String> and a < b"])
    assert vis == "use Vec<String> and a < b"
    assert think == ""


def test_thinking_is_never_serialized_into_history():
    """The whole point: history must not carry the reasoning."""
    turn = AssistantTurn(text="the answer", thinking="lots of private reasoning")
    msg = turn.to_message()
    assert msg["content"] == "the answer"
    assert "reasoning" not in str(msg)
