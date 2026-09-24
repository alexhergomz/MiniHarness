"""Resuming a session. The interesting cases are the ugly ones: a session that
died mid-turn, and continuing to write after a resume."""
import json

import pytest

from miniharness import loop, session


@pytest.fixture(autouse=True)
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSIONS", tmp_path / "sessions")


def _write(sid, messages):
    session.append(sid, {"messages": messages})


CRASHED = [
    {"role": "user", "content": "fix the bug"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "Read", "arguments": "{}"}}]},
    # process died here — no tool response was ever written
]


def test_round_trip(tmp_path):
    _write("s1", [{"role": "user", "content": "hi"}])
    assert session.load("s1")[-1]["messages"][0]["content"] == "hi"


def test_a_session_that_died_mid_tool_call_is_repaired_on_resume():
    """The transcript on disk is only repaired if the process caught
    KeyboardInterrupt. A SIGKILL, a crash, or a closed laptop leaves a dangling
    tool_call, and resuming it 400s on the very next request."""
    _write("crashed", CRASHED)
    st = loop.State(messages=session.load("crashed")[-1]["messages"])
    assert loop.repair_history(st.messages) == 1
    ids = [m.get("tool_call_id") for m in st.messages if m["role"] == "tool"]
    assert "call_1" in ids, "dangling call must get a tool response"


def test_resumed_history_is_valid_for_the_next_request():
    _write("crashed", CRASHED)
    st = loop.State(messages=session.load("crashed")[-1]["messages"])
    loop.repair_history(st.messages)
    pending = []
    for m in st.messages:
        if m["role"] == "assistant":
            assert not pending, "unanswered tool calls before a new assistant turn"
            pending = [t["id"] for t in m.get("tool_calls", [])]
        elif m["role"] == "tool":
            pending.remove(m["tool_call_id"])
    assert not pending


def test_latest_picks_the_newest(tmp_path):
    import os, time
    _write("older", [{"role": "user", "content": "a"}])
    _write("newer", [{"role": "user", "content": "b"}])
    p = session.path_for("newer")
    os.utime(p, (time.time() + 10, time.time() + 10))
    assert session.latest() == "newer"


def test_corrupt_lines_are_skipped_not_fatal():
    _write("mixed", [{"role": "user", "content": "good"}])
    with open(session.path_for("mixed"), "a") as f:
        f.write("{not json at all\n")
    _write("mixed", [{"role": "user", "content": "later"}])
    recs = session.load("mixed")
    assert len(recs) == 2 and recs[-1]["messages"][0]["content"] == "later"


def test_resume_continues_writing_to_the_same_session():
    """After /resume, further turns must append to the resumed transcript. If
    they go to the new session id instead, the conversation silently splits
    across two files and the resumed one stops growing."""
    from miniharness import __main__ as cli
    _write("old", [{"role": "user", "content": "earlier work"}])
    st = loop.State()
    st.session_id = "fresh"
    cli.handle_command("/resume old", st, {"_cwd": "."}, None)
    assert st.session_id == "old", "resume did not rebind the session id"
    assert st.messages[0]["content"] == "earlier work"


def test_resuming_a_crashed_session_repairs_it_before_any_request():
    """End to end through the command, not just the helper: a transcript that
    died mid-tool-call must be usable the moment it is resumed."""
    from miniharness import __main__ as cli
    _write("crashed", CRASHED)
    st = loop.State()
    cli.handle_command("/resume crashed", st, {"_cwd": "."}, None)
    assert st.session_id == "crashed"
    assert loop.repair_history(st.messages) == 0, "resume left history unrepaired"
    assert any(m["role"] == "tool" and m["tool_call_id"] == "call_1"
               for m in st.messages)


def test_a_resumed_session_remembers_what_it_read(tmp_path):
    """The tracker is in memory and starts empty, so after /resume a file read
    last session counted as unread — and the read-before-edit rule fired on a
    file whose contents were in the model's context. Measured live: the first
    Edit after a resume is refused, costing a turn to re-read something visible.
    """
    import json

    from miniharness import context, tools

    f = tmp_path / "calc.py"
    f.write_text("def add(a, b):\n    return a - b\n")
    messages = [
        {"role": "user", "content": "look at calc.py"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "Read",
                          "arguments": json.dumps({"file_path": "calc.py"})}}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": "     1\tdef add(a, b):\n     2\t    return a - b"},
    ]
    tracker = context.FileTracker.from_messages(messages, str(tmp_path))
    assert tracker.has_read(str(f))

    out = tools.dispatch("Edit", {"file_path": "calc.py",
                                  "old_string": "return a - b",
                                  "new_string": "return a + b"},
                         {"_cwd": str(tmp_path)}, tracker)
    assert not out.startswith("Error"), out
    assert "a + b" in f.read_text()


def test_a_failed_read_does_not_count_as_having_read(tmp_path):
    import json

    from miniharness import context
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "Read",
                          "arguments": json.dumps({"file_path": "missing.py"})}}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": "Error: missing.py does not exist"},
    ]
    tracker = context.FileTracker.from_messages(messages, str(tmp_path))
    assert not tracker.has_read(str(tmp_path / "missing.py"))


def test_only_reads_count_not_other_tools(tmp_path):
    import json

    from miniharness import context
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "Grep",
                          "arguments": json.dumps({"pattern": "x",
                                                   "file_path": "a.py"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.py:1:x = 1"},
    ]
    tracker = context.FileTracker.from_messages(messages, str(tmp_path))
    assert not tracker.has_read(str(tmp_path / "a.py")), \
        "a grep hit is not the same as having read the file"


# ── Saving as it happens ────────────────────────────────────────────────────
def test_every_step_is_on_disk_before_the_turn_ends():
    """A session used to be written once per turn, so a crash or a sleeping
    laptop mid-turn lost everything since the user's last message."""
    from miniharness import session
    st = loop.State(messages=[{"role": "user", "content": "go"}], session_id="inc")
    session.save_point(st, "/w")
    for i in range(3):
        st.messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}]})
        st.messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"out{i}"})
        st.ledger.append((f"c{i}", "Bash", {}, f"out{i}"))
        session.save_point(st, "/w")
        assert session.restore("inc")["messages"] == st.messages
    recs = session.load("inc")
    assert "meta" in recs[0] and recs[0]["meta"]["cwd"] == "/w"
    assert sum("add" in r for r in recs) == 3, "appends should be written as appends"
    assert len(session.restore("inc")["ledger"]) == 3


def test_a_rewritten_history_is_saved_as_a_snapshot():
    """Compaction replaces old messages. Appending after that would replay a
    history that never existed."""
    from miniharness import session
    st = loop.State(messages=[{"role": "user", "content": f"m{i}"} for i in range(6)],
                    session_id="snap")
    session.save_point(st, "/w")
    st.messages = [{"role": "user", "content": "Notes from the earlier part of this session:\n…"},
                   {"role": "user", "content": "m5"}]
    session.save_point(st, "/w")
    assert session.restore("snap")["messages"] == st.messages
    assert "messages" in session.load("snap")[-1]


def test_old_session_files_still_load():
    from miniharness import session
    session.append("legacy", {"messages": [{"role": "user", "content": "a"}]})
    session.append("legacy", {"messages": [{"role": "user", "content": "a"},
                                           {"role": "assistant", "content": "b"}]})
    assert [m["content"] for m in session.restore("legacy")["messages"]] == ["a", "b"]


def test_an_interrupted_turn_is_recognised():
    from miniharness import session
    user = {"role": "user", "content": "go"}
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}
    result = {"role": "tool", "tool_call_id": "c1", "content": "x"}
    done = {"role": "assistant", "content": "finished"}
    assert session.unfinished([user])
    assert session.unfinished([user, call])
    assert session.unfinished([user, call, result])
    assert not session.unfinished([user, call, result, done])
    assert not session.unfinished([])


def test_continue_picks_the_latest_session_in_this_directory():
    from miniharness import session
    for sid, cwd in (("a-1", "/proj/a"), ("b-1", "/proj/b")):
        st = loop.State(messages=[{"role": "user", "content": sid}], session_id=sid)
        session.save_point(st, cwd)
    assert session.latest_for("/proj/a") == "a-1"
    assert session.latest_for("/proj/b") == "b-1"
    assert session.latest_for("/proj/c") is None
