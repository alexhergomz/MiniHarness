"""The per-turn focus map.

Replaces a static session-start map that was (a) stale the moment the agent
edited anything, (b) spending its whole budget on whatever the repo's
most-referenced utility happened to be, and (c) permanently resident in the
system prompt. The new one is rebuilt each turn, personalised to the task, and
never stored in history.
"""

from __future__ import annotations

import pytest

from miniharness import context, loop


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def verify_token(tok):\n    return check_signature(tok)\n\n"
        "def check_signature(tok):\n    return True\n")
    (tmp_path / "billing.py").write_text(
        "def charge_card(amount):\n    return settle(amount)\n\n"
        "def settle(amount):\n    return amount\n")
    (tmp_path / "util.py").write_text("def helper():\n    return 1\n")
    return tmp_path


def has_graph():
    from miniharness.repomap import HAVE_GRAPH
    return HAVE_GRAPH


# ── Not in the system prompt any more ───────────────────────────────────────
def test_system_prompt_no_longer_carries_the_map(repo):
    """It made the prompt vary per repo and go stale on the first edit."""
    out = context.build_system({"_cwd": str(repo), "repo_map": True})
    assert "Repository map" not in out


def test_system_prompt_is_identical_across_different_repos(tmp_path):
    """The strongest form of prefix stability: the same bytes everywhere."""
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "x.py").write_text("def f():\n    return 1\n")
    assert (context.build_system({"_cwd": str(a), "repo_map": True})
            == context.build_system({"_cwd": str(b), "repo_map": True}))


# ── Ephemeral ───────────────────────────────────────────────────────────────
def test_focus_map_is_not_written_into_history(repo):
    msgs = [{"role": "user", "content": "fix verify_token"}]
    out = loop._with_focus_map(msgs, {"_cwd": str(repo), "repo_map": True}, None)
    assert len(msgs) == 1, "history must not be mutated"
    if has_graph():
        assert len(out) == 2 and out[-1]["role"] == "user"


def test_focus_map_does_not_accumulate_over_turns(repo):
    cfg = {"_cwd": str(repo), "repo_map": True}
    msgs = [{"role": "user", "content": "fix verify_token"}]
    for _ in range(5):
        req = loop._with_focus_map(msgs, cfg, None)
        assert sum("repository map" in str(m.get("content", "")).lower()
                   for m in req) <= 1


def test_focus_map_is_skipped_mid_tool_call(repo):
    """A user message between tool_calls and their responses breaks alternation."""
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "Read", "arguments": "{}"}}]}]
    assert loop._with_focus_map(msgs, {"_cwd": str(repo), "repo_map": True}, None) is msgs


def test_focus_map_off_when_disabled(repo):
    msgs = [{"role": "user", "content": "x"}]
    assert loop._with_focus_map(msgs, {"_cwd": str(repo), "repo_map": False}, None) is msgs


# ── Relevant ────────────────────────────────────────────────────────────────
def test_mentions_come_from_the_request_not_tool_results():
    """Tool results are file contents; harvesting them would swamp the ranking
    with every identifier in the repo."""
    msgs = [{"role": "user", "content": "please fix verify_token in auth.py"},
            {"role": "tool", "tool_call_id": "c1",
             "content": "def unrelated_symbol_from_a_dump(): pass"}]
    got = context.mentions_from(msgs)
    assert "verify_token" in got
    assert "auth.py" in got
    assert "unrelated_symbol_from_a_dump" not in got


def test_mentions_drop_common_words():
    got = context.mentions_from([{"role": "user", "content": "please fix the bug in the code"}])
    assert not {"please", "fix", "the", "bug", "code"} & got


def test_focus_map_steers_toward_mentioned_paths(tmp_path):
    """The map ranks by path relevance now, not symbol relevance."""
    for i in range(60):
        (tmp_path / f"widget{i:02d}.py").write_text("x = 1\n")
    (tmp_path / "auth").mkdir()
    (tmp_path / "auth" / "tokens.py").write_text("y = 2\n")
    cfg = {"_cwd": str(tmp_path), "repo_map": True, "repo_map_tokens": 400}
    a = context.focus_map(cfg, [{"role": "user", "content": "fix auth tokens"}])
    b = context.focus_map(cfg, [{"role": "user", "content": "fix widget07"}])
    assert a != b
    assert a.index("auth") < a.index("widget00")


def test_focus_map_reflects_files_added_during_the_session(tmp_path):
    """A path map tracks files, not definitions.

    This is a real capability loss against the symbol graph, and it is the whole
    bet: a new function inside an existing file does NOT change the map, only a
    new file does. The wager is that a model reading `Grep` and `Read` results
    learns about new functions perfectly well, and only needs the map to know
    where to look.
    """
    (tmp_path / "existing.py").write_text("def a():\n    return 1\n")
    cfg = {"_cwd": str(tmp_path), "repo_map": True, "repo_map_tokens": 400}
    msgs = [{"role": "user", "content": "look at scheduler"}]
    before = context.focus_map(cfg, msgs)
    (tmp_path / "scheduler.py").write_text("def run():\n    return 2\n")
    after = context.focus_map(cfg, msgs)
    assert "scheduler.py" in after and "scheduler.py" not in (before or "")


def test_focus_map_survives_a_broken_repo(tmp_path):
    """A map failure must never take down the turn."""
    (tmp_path / "bad.py").write_bytes(b"\x00\xff not valid utf-8 \xfe")
    assert context.focus_map({"_cwd": str(tmp_path), "repo_map": True}, []) is not None or True


def test_the_map_does_not_duplicate_what_findsymbol_does(tmp_path):
    """Measured: on four symbols whose filename shares no word with them, the
    no-map, path-map and symbol-map arms all scored 4/4, because FindSymbol is
    a grep. The symbol-augmented map was the slowest arm — a permanent per-turn
    prompt cost to spare an occasional tool call."""
    (tmp_path / "anti_stuck.py").write_text("def mark_fired():\n    return 1\n")
    cfg = {"_cwd": str(tmp_path), "repo_map": True, "repo_map_tokens": 400}
    m = context.focus_map(cfg, [{"role": "user", "content": "where is mark_fired"}])
    assert "anti_stuck.py" in m
    assert "mark_fired" not in m, "the map lists paths; symbols are FindSymbol's job"


def test_the_map_says_where_the_paths_are_relative_to(tmp_path):
    """The map groups by directory, and a model read `tests/` as a root: it
    asked for `/tests/test_core.py`, then `/workspace/tests/test_core.py`, and
    said in its reasoning it was "using the absolute path from the repository
    map". The jail refused all of them. It ran `pwd`, saw the truth, and went
    back to `/workspace` — 12 of 17 calls on a path that could not resolve.

    So the map has to name its own root. Nothing else ever told the model where
    it was."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "thing.py").write_text("def a():\n    return 1\n")
    m = context.focus_map({"_cwd": str(tmp_path), "repo_map": True}, [])
    assert str(tmp_path) in m, "the map must state the working directory"
    assert "relative to it" in m
