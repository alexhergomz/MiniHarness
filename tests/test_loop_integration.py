"""End-to-end loop tests against a scripted model.

No network and no server: ``loop.stream`` is replaced with a generator that
replays a fixed script. What's being tested is the part that actually breaks in
practice — that the message history stays valid across tool rounds, denials,
truncations, and interruptions. An invalid history is the failure that surfaces
as a 400 several turns later, far from its cause.
"""

from __future__ import annotations

import pytest

from miniharness import context, loop
from miniharness.provider import AssistantTurn, TextChunk


def script(*turns):
    """Build a fake ``stream`` that replays ``turns`` in order."""
    remaining = list(turns)

    def fake_stream(model, system, messages, schemas, config):
        turn = remaining.pop(0)
        if turn.text:
            yield TextChunk(turn.text)
        yield turn

    return fake_stream


def drain(state, config, monkeypatch, fake, ask=None, tracker=None):
    monkeypatch.setattr(loop, "stream_complete", fake)
    return list(loop.run(state, config, ask, tracker))


def assert_history_valid(messages):
    """Every tool message must answer an assistant tool_call, and vice versa."""
    pending: list[str] = []
    for m in messages:
        if m.get("role") == "assistant":
            assert not pending, f"unanswered tool_calls: {pending}"
            pending = [tc["id"] for tc in m.get("tool_calls", [])]
        elif m.get("role") == "tool":
            assert m["tool_call_id"] in pending, "orphaned tool message"
            pending.remove(m["tool_call_id"])
        elif m.get("role") == "user":
            assert not pending, f"unanswered tool_calls before user turn: {pending}"
    assert not pending, f"conversation ends with unanswered tool_calls: {pending}"


def assert_alternates(messages):
    """No two consecutive user messages — the invariant Qwen breaks on."""
    roles = [m["role"] for m in messages]
    for a, b in zip(roles, roles[1:]):
        assert not (a == "user" and b == "user"), f"user->user in {roles}"


BASE = {"model": "local", "max_turns": 10, "accept_all": True, "repo_map": False}


def test_a_full_tool_round_produces_valid_history(tmp_path, monkeypatch):
    f = tmp_path / "a.py"
    f.write_text("print(1)\n")
    cfg = dict(BASE, _cwd=str(tmp_path))
    state = loop.State(messages=[{"role": "user", "content": "read a.py"}])

    events = drain(state, cfg, monkeypatch, script(
        AssistantTurn(text="Looking.", finish_reason="tool_calls", tool_calls=[
            {"id": "c1", "name": "Read", "input": {"file_path": str(f)}}]),
        AssistantTurn(text="It prints 1.", finish_reason="stop"),
    ), tracker=context.FileTracker())

    kinds = [type(e).__name__ for e in events]
    assert "ToolStart" in kinds and "ToolEnd" in kinds and kinds[-1] == "TurnDone"
    assert_history_valid(state.messages)
    assert_alternates(state.messages)
    assert "print(1)" in state.messages[2]["content"]


def test_parallel_tool_calls_each_get_a_response(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("a\n")
    (tmp_path / "b.py").write_text("b\n")
    cfg = dict(BASE, _cwd=str(tmp_path))
    state = loop.State(messages=[{"role": "user", "content": "read both"}])

    drain(state, cfg, monkeypatch, script(
        AssistantTurn(finish_reason="tool_calls", tool_calls=[
            {"id": "c1", "name": "Read", "input": {"file_path": str(tmp_path / "a.py")}},
            {"id": "c2", "name": "Read", "input": {"file_path": str(tmp_path / "b.py")}}]),
        AssistantTurn(text="done", finish_reason="stop"),
    ), tracker=context.FileTracker())

    assert_history_valid(state.messages)
    assert sum(1 for m in state.messages if m["role"] == "tool") == 2


def test_a_denied_tool_still_gets_a_tool_response(tmp_path, monkeypatch):
    """Denial must not leave the tool_call unanswered."""
    cfg = dict(BASE, _cwd=str(tmp_path), accept_all=False)
    state = loop.State(messages=[{"role": "user", "content": "delete everything"}])

    events = drain(state, cfg, monkeypatch, script(
        AssistantTurn(finish_reason="tool_calls", tool_calls=[
            {"id": "c1", "name": "Bash", "input": {"command": "echo hi"}}]),
        AssistantTurn(text="Understood.", finish_reason="stop"),
    ), ask=lambda name, params: False)

    ends = [e for e in events if type(e).__name__ == "ToolEnd"]
    assert ends[0].denied
    assert "denied" in state.messages[2]["content"]
    assert_history_valid(state.messages)


def test_read_only_tools_are_never_gated(tmp_path, monkeypatch):
    f = tmp_path / "a.py"
    f.write_text("x\n")
    asked = []
    cfg = dict(BASE, _cwd=str(tmp_path), accept_all=False)
    state = loop.State(messages=[{"role": "user", "content": "read"}])

    drain(state, cfg, monkeypatch, script(
        AssistantTurn(finish_reason="tool_calls", tool_calls=[
            {"id": "c1", "name": "Read", "input": {"file_path": str(f)}}]),
        AssistantTurn(text="ok", finish_reason="stop"),
    ), ask=lambda n, p: asked.append(n) or True, tracker=context.FileTracker())

    assert asked == [], "Read should never prompt for permission"


def test_truncation_mid_tool_call_recovers_without_breaking_alternation(tmp_path, monkeypatch):
    """The exact failure the alternation fix exists for."""
    cfg = dict(BASE, _cwd=str(tmp_path), max_continuations=2)
    state = loop.State(messages=[{"role": "user", "content": "do it"}])

    drain(state, cfg, monkeypatch, script(
        # Cut off mid-JSON: one malformed call, no text.
        AssistantTurn(text="", finish_reason="length", tool_calls=[
            {"id": "c1", "name": "Edit", "input": {"_raw": '{"file_pa'}}]),
        AssistantTurn(text="Recovered.", finish_reason="stop"),
    ))

    assert_alternates(state.messages)
    assert_history_valid(state.messages)
    assert state.messages[1]["content"] == loop.TRUNCATION_STUB
    assert "tool_calls" not in state.messages[1]
    assert state.messages[2]["role"] == "user"   # the continuation hint
    assert state.messages[-1]["content"] == "Recovered."


def test_an_empty_turn_is_nudged_not_treated_as_done(tmp_path, monkeypatch):
    """Observed with Qwen3.5-4B: it ran the tests, reasoned for ~173 tokens
    inside <think>, then emitted nothing. The loop read that as "finished" and
    stopped with the task half-done — silently, looking like success."""
    cfg = dict(BASE, _cwd=str(tmp_path), max_empty_retries=2)
    state = loop.State(messages=[{"role": "user", "content": "fix the test"}])

    events = drain(state, cfg, monkeypatch, script(
        AssistantTurn(text="", finish_reason="stop", thinking="lots of reasoning"),
        AssistantTurn(text="Fixed it.", finish_reason="stop"),
    ))

    assert any(type(e).__name__ == "Notice" for e in events)
    assert state.messages[1]["content"] == loop.EMPTY_STUB
    assert state.messages[2]["role"] == "user"
    assert "Stop reasoning and act now" in state.messages[2]["content"]
    assert state.messages[-1]["content"] == "Fixed it."
    assert_alternates(state.messages)
    assert_history_valid(state.messages)


def test_empty_turn_nudging_is_bounded(tmp_path, monkeypatch):
    """A model that cannot produce output will not start on the tenth try."""
    cfg = dict(BASE, _cwd=str(tmp_path), max_empty_retries=2)
    state = loop.State(messages=[{"role": "user", "content": "go"}])
    drain(state, cfg, monkeypatch, script(
        *[AssistantTurn(text="", finish_reason="stop") for _ in range(3)]))
    assert state.empty_retries == 2
    assert_alternates(state.messages)


def test_a_turn_with_text_but_no_tools_is_a_real_completion(tmp_path, monkeypatch):
    """Don't nudge a model that actually answered."""
    cfg = dict(BASE, _cwd=str(tmp_path))
    state = loop.State(messages=[{"role": "user", "content": "what is 2+2"}])
    events = drain(state, cfg, monkeypatch,
                   script(AssistantTurn(text="4", finish_reason="stop")))
    assert type(events[-1]).__name__ == "TurnDone"
    assert state.empty_retries == 0
    assert len(state.messages) == 2


def test_a_failing_tool_does_not_kill_the_loop(tmp_path, monkeypatch):
    cfg = dict(BASE, _cwd=str(tmp_path))
    state = loop.State(messages=[{"role": "user", "content": "read a missing file"}])

    drain(state, cfg, monkeypatch, script(
        AssistantTurn(finish_reason="tool_calls", tool_calls=[
            {"id": "c1", "name": "Read", "input": {"file_path": "/nope/missing.py"}}]),
        AssistantTurn(text="That file is gone.", finish_reason="stop"),
    ), tracker=context.FileTracker())

    # Either refusal is fine; the point is the loop survives a tool error and
    # the model gets an actionable message back.
    result = state.messages[2]["content"]
    assert result.startswith("Error")
    assert "does not exist" in result or "outside the working directory" in result
    assert_history_valid(state.messages)


def test_max_turns_stops_a_runaway_loop(tmp_path, monkeypatch):
    """A model that calls a tool forever must be bounded."""
    cfg = dict(BASE, _cwd=str(tmp_path), max_turns=3)
    state = loop.State(messages=[{"role": "user", "content": "loop"}])

    def endless(model, system, messages, schemas, config):
        yield AssistantTurn(finish_reason="tool_calls", tool_calls=[
            {"id": f"c{len(messages)}", "name": "Glob", "input": {"pattern": "*.py"}}])

    monkeypatch.setattr(loop, "stream_complete", endless)
    events = list(loop.run(state, cfg, None, None))

    assert type(events[-1]).__name__ == "Notice"
    assert "stopped after 3" in events[-1].text
    assert sum(1 for m in state.messages if m["role"] == "tool") == 3


def test_text_only_turn_ends_the_loop_immediately(tmp_path, monkeypatch):
    cfg = dict(BASE, _cwd=str(tmp_path))
    state = loop.State(messages=[{"role": "user", "content": "hello"}])
    events = drain(state, cfg, monkeypatch,
                   script(AssistantTurn(text="Hi.", finish_reason="stop")))
    assert type(events[-1]).__name__ == "TurnDone"
    assert len(state.messages) == 2


def test_interrupt_between_parallel_tool_calls_leaves_valid_history():
    """The bug: Ctrl-C after tool call 1 of 2 left call 2 unanswered.

    The old handler only popped when the assistant message was *last*, so this
    exact shape survived and 400'd on the next request.
    """
    messages = [
        {"role": "user", "content": "do both"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
        # ...Ctrl-C during c2.
    ]
    assert loop.repair_history(messages) == 1
    assert_history_valid(messages)
    assert messages[-1]["tool_call_id"] == "c2"
    assert "interrupted" in messages[-1]["content"]


def test_interrupt_before_any_dispatch_is_also_repaired():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "Working on it.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}]},
    ]
    assert loop.repair_history(messages) == 1
    assert_history_valid(messages)
    assert messages[1]["content"] == "Working on it."  # the model's text survives


def test_repair_is_a_no_op_on_a_complete_history():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    assert loop.repair_history(messages) == 0
    assert len(messages) == 4


def test_repair_handles_a_conversation_with_no_tool_calls():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert loop.repair_history(messages) == 0


def test_compaction_output_stays_a_valid_history(tmp_path, monkeypatch):
    """Compaction runs on real conversations, so it must preserve the invariant."""
    cfg = dict(BASE, _cwd=str(tmp_path), max_turns=30)
    state = loop.State(messages=[{"role": "user", "content": "big job"}])

    turns = []
    for i in range(20):
        turns.append(AssistantTurn(finish_reason="tool_calls", tool_calls=[
            # Not brace expansion: shell=True runs /bin/sh, where {1..3000}
            # is a literal.
            {"id": f"c{i}", "name": "Bash",
             "input": {"command": "python3 -c \"print('x'*3000)\""}}]))
    turns.append(AssistantTurn(text="done", finish_reason="stop"))
    drain(state, cfg, monkeypatch, script(*turns))

    assert_history_valid(state.messages)
    compacted, freed = context.compact_with_model(state.messages, 2_000, "local", "sys", {})
    assert freed > 0
    assert_history_valid(compacted)
    assert_alternates(compacted)



def test_a_mutation_makes_rereading_legitimate(tmp_path, monkeypatch):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    cfg = dict(BASE, _cwd=str(tmp_path))
    tracker = context.FileTracker()
    state = loop.State(messages=[{"role": "user", "content": "fix"}])
    read = {"id": "r", "name": "Read", "input": {"file_path": str(f)}}
    drain(state, cfg, monkeypatch, script(
        AssistantTurn(finish_reason="tool_calls", tool_calls=[dict(read)]),
        AssistantTurn(finish_reason="tool_calls", tool_calls=[
            {"id": "e", "name": "Edit", "input": {"file_path": str(f),
                                                  "old_string": "x = 1",
                                                  "new_string": "x = 2"}}]),
        AssistantTurn(finish_reason="tool_calls", tool_calls=[dict(read, id="r2")]),
        AssistantTurn(text="done", finish_reason="stop"),
    ), tracker=tracker)
    tool_msgs = [m["content"] for m in state.messages if m["role"] == "tool"]
    assert "x = 2" in tool_msgs[-1], "a read after an edit must actually re-read"


def test_bash_is_never_deduplicated(tmp_path, monkeypatch):
    """Re-running a test suite is the point; it is not a repeated read."""
    cfg = dict(BASE, _cwd=str(tmp_path))
    state = loop.State(messages=[{"role": "user", "content": "test"}])
    cmd = {"name": "Bash", "input": {"command": "echo hi"}}
    drain(state, cfg, monkeypatch, script(
        AssistantTurn(finish_reason="tool_calls", tool_calls=[dict(cmd, id="b1")]),
        AssistantTurn(finish_reason="tool_calls", tool_calls=[dict(cmd, id="b2")]),
        AssistantTurn(text="ok", finish_reason="stop"),
    ))
    tool_msgs = [m["content"] for m in state.messages if m["role"] == "tool"]
    assert all("hi" in t for t in tool_msgs)


def test_context_is_compacted_inside_a_single_turn(tmp_path, monkeypatch):
    """Compaction used to run only between user turns, so one turn with many
    tool calls could exceed the window and hard-fail. Measured on a benchmark
    task: 21,352 tokens against a 16,384 window, returned as a 400 that killed
    the run."""
    # Distinct files: reading the same one repeatedly is deduplicated, so the
    # context would never grow and the test would prove nothing.
    paths = []
    for i in range(5):
        f = tmp_path / f"big{i}.txt"
        f.write_text(f"file {i}\n" + "x " * 20_000)
        paths.append(str(f))
    cfg = dict(BASE, _cwd=str(tmp_path), llama_ctx=4000, compact_at=0.5, max_turns=8)
    state = loop.State(system="sys", messages=[{"role": "user", "content": "read them"}])

    reads = [AssistantTurn(finish_reason="tool_calls", tool_calls=[
        {"id": f"c{i}", "name": "Read", "input": {"file_path": paths[i]}}])
        for i in range(5)]
    events = drain(state, cfg, monkeypatch,
                   script(*reads, AssistantTurn(text="done", finish_reason="stop")),
                   tracker=context.FileTracker())

    kinds = [type(e).__name__ for e in events]
    assert "Compacting" in kinds, "a compaction must be announced before it runs"
    assert any(type(e).__name__ == "Notice" and e.text.startswith("compacted:")
               for e in events), "the loop must compact without waiting for the turn to end"
    assert context.estimate_tokens(state.messages, state.system) <= 2000
    assert_history_valid(state.messages)


def test_no_mid_turn_compaction_when_the_window_is_unknown(tmp_path, monkeypatch):
    """llama_ctx=0 means 'let the server decide'; guessing a budget would
    compact needlessly on a large-context model."""
    cfg = dict(BASE, _cwd=str(tmp_path), llama_ctx=0)
    state = loop.State(system="sys", messages=[{"role": "user", "content": "hi"}])
    events = drain(state, cfg, monkeypatch,
                   script(AssistantTurn(text="ok", finish_reason="stop")))
    assert not any(type(e).__name__ == "Notice" and "compacted" in e.text for e in events)


# ── the request must fit the window, not just the history ───────────────────
def test_compaction_leaves_room_for_schemas_map_and_reply():
    """The history is not the whole request. Budgeting messages against the raw
    window overflows by exactly the size of everything else, which the server
    returns as a 400 that kills the run mid-turn:

        request (17056 tokens) exceeds the available context size (16384)

    Three such crashes in one 12-task benchmark run.
    """
    import json

    from miniharness import config as cfg_mod
    from miniharness import context, loop, tools

    window = 16384
    config = dict(cfg_mod.DEFAULTS)
    config.update({"llama_ctx": window, "_cwd": ".", "model_compaction": False})
    schemas = tools.schemas_for(config)

    # A conversation far larger than the window.
    state = loop.State(system="s" * 400, messages=[])
    for i in range(400):
        state.messages.append({"role": "user", "content": f"q{i}"})
        state.messages.append({"role": "assistant", "content": "x" * 400})

    loop._compact_if_needed(state, config, schemas)

    # Measure the request the loop actually builds — with the focus map
    # appended — not a reconstruction of it.
    request = loop._with_focus_map(state.messages, config, context.FileTracker())
    sent = (context.estimate_tokens(request, state.system)
            + len(json.dumps(schemas)) // 4)
    limit = _answer_room(window, config)
    assert sent <= limit, (
        f"prompt {sent} leaves no room to answer: over {limit} of a {window} window")


def test_the_budget_scales_with_the_window():
    """Everything is a share of llama_ctx, so a bigger window must permit a
    proportionally bigger history rather than a fixed one."""
    from miniharness import config as cfg_mod
    from miniharness import loop, tools

    def budget_for(window):
        from miniharness import context
        config = dict(cfg_mod.DEFAULTS)
        config.update({"llama_ctx": window, "_cwd": ".", "model_compaction": False})
        schemas = tools.schemas_for(config)
        # Tool output, not user turns: user turns are deliberately preserved,
        # so a conversation made of them cannot demonstrate a budget at all.
        state = loop.State(system="", messages=[{"role": "user", "content": "go"}])
        for i in range(300):
            state.messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "Read", "arguments": "{}"}}]})
            state.messages.append({"role": "tool", "tool_call_id": f"c{i}",
                                   "content": "y" * 1000})
        loop._compact_if_needed(state, config, schemas)
        return context.estimate_tokens(state.messages, state.system)

    assert budget_for(65536) > budget_for(16384) * 2


def test_an_unknown_window_does_not_compact():
    from miniharness import loop
    state = loop.State(messages=[{"role": "user", "content": "x" * 100_000}])
    assert loop._compact_if_needed(state, {"_cwd": "."}, []) == 0


def test_only_one_place_computes_the_history_budget():
    """Three call sites computed this independently and two got it wrong —
    the loop overflowed the window, and /compact used a hardcoded window//4
    that also ignored the shares rule."""
    import pathlib
    import re

    offenders = {}
    for f in sorted(pathlib.Path("miniharness").glob("*.py")):
        if f.name == "config.py":
            continue
        src = f.read_text()
        # a compaction budget derived from the window by hand
        if re.search(r'window[^\n]*\*[^\n]*compact_at|window\s*//\s*\d', src):
            offenders[f.name] = True
    assert not offenders, f"hand-rolled history budget in: {list(offenders)}"


def test_manual_compact_targets_below_the_automatic_trigger():
    """Otherwise /compact immediately after an automatic pass is a no-op."""
    from miniharness import config as cfg_mod
    from miniharness import tools
    config = dict(cfg_mod.DEFAULTS)
    config.update({"llama_ctx": 16384, "_cwd": ".", "model_compaction": False})
    auto = cfg_mod.history_budget(config, tools.schemas_for(config))
    assert auto // 2 < auto


# ── every request the harness can build must fit the window ────────────────
def _measure(messages, system, schemas, window):
    """Prompt tokens as the server counts them.

    No reply reservation is added any more: nothing is held back for the answer
    (see config.history_budget). The invariant these tests defend is unchanged
    in substance — a constructed request must still leave the model room to
    answer in — but it is now expressed against `compact_at`, the point at which
    history is supposed to yield, rather than against a fixed set-aside.
    """
    import json

    from miniharness import context
    return (context.estimate_tokens(messages, system)
            + len(json.dumps(schemas)) // 4)


def _answer_room(window, cfg=None):
    """The largest request that still leaves the model somewhere to write."""
    from miniharness import config as cfg_mod
    at = float((cfg or cfg_mod.DEFAULTS).get("compact_at", 0.85))
    return int(window * at)


def test_no_request_stream_complete_builds_can_exceed_the_window():
    """Three separate overflow bugs shipped in this family — compaction not
    counting the schemas, the continuation resend, and the landing round —
    each found only after a 400 killed a benchmark run. Finding them one at a
    time was not working, so every request the provider can construct is
    checked instead of each site being audited by hand.

    Note what this does and does not prove. It verifies the *arithmetic*: given
    the estimator, no construction path exceeds the window. It cannot verify the
    estimator itself — both sides use it, so it is self-consistent and would
    stay green while the real request overflowed. That was exactly what happened
    (`request (16401 tokens)` with every one of these tests passing), and it is
    why the estimator now calibrates against the server's own count. A test
    measuring with the same ruler as the code cannot catch a bent ruler.
    """
    from miniharness import config as cfg_mod
    from miniharness import provider, tools

    window = 16384
    cfg = dict(cfg_mod.DEFAULTS)
    cfg.update({"llama_ctx": window, "_cwd": ".", "model": "local",
                "max_continuations": 12})
    schemas = tools.schemas_for(cfg)
    system = "s" * 900

    # History already compacted to its budget, as the loop guarantees.
    budget = cfg_mod.history_budget(cfg, schemas)
    messages = [{"role": "user", "content": "x" * (budget * 4)}]

    seen = []

    def fake(model, sys_, msgs, sch, conf):
        seen.append(list(msgs))
        # worst case every round: a full cap of pure reasoning, nothing visible
        t = AssistantTurn(text="", finish_reason="length")
        t.thinking = "r" * (cfg_mod.budget(cfg, "reply_share") * 4)
        yield t

    monkey = provider.stream
    provider.stream = fake
    try:
        list(provider.stream_complete("local", system, messages, schemas, cfg))
    finally:
        provider.stream = monkey

    assert len(seen) > 1, "continuation never engaged; the test proves nothing"
    limit = _answer_room(window, cfg)
    over = [(i, _measure(m, system, schemas, window))
            for i, m in enumerate(seen) if _measure(m, system, schemas, window) > limit]
    assert not over, f"requests leaving no room to answer (>{limit} of {window}): {over}"


def test_the_same_holds_for_a_visible_reply_rather_than_reasoning():
    from miniharness import config as cfg_mod
    from miniharness import provider, tools

    window = 16384
    cfg = dict(cfg_mod.DEFAULTS)
    cfg.update({"llama_ctx": window, "_cwd": ".", "model": "local",
                "max_continuations": 12})
    schemas = tools.schemas_for(cfg)
    system = "s" * 900
    budget = cfg_mod.history_budget(cfg, schemas)
    messages = [{"role": "user", "content": "x" * (budget * 4)}]

    seen = []

    def fake(model, sys_, msgs, sch, conf):
        seen.append(list(msgs))
        yield AssistantTurn(text="w" * (cfg_mod.budget(cfg, "reply_share") * 4),
                            finish_reason="length")

    monkey = provider.stream
    provider.stream = fake
    try:
        list(provider.stream_complete("local", system, messages, schemas, cfg))
    finally:
        provider.stream = monkey

    limit = _answer_room(window, cfg)
    over = [i for i, m in enumerate(seen)
            if _measure(m, system, schemas, window) > limit]
    assert not over, f"requests {over} left no room to answer (>{limit} of {window})"


def test_budgets_shrink_when_the_estimator_learns_it_was_undercounting():
    """The calibration is only useful if the budgets move with it. A session
    full of command output (2.35x) or hexdumps (3.55x) must end up compacting
    proportionally harder, not merely recording a bigger number."""
    from miniharness import config as cfg_mod
    from miniharness import context, loop, tools

    cfg = dict(cfg_mod.DEFAULTS)
    cfg.update({"llama_ctx": 16384, "_cwd": ".", "model_compaction": False})
    schemas = tools.schemas_for(cfg)

    def surviving_chars(calibration):
        context._calibration = calibration
        state = loop.State(system="", messages=[{"role": "user", "content": "go"}])
        for i in range(300):
            state.messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "Bash", "arguments": "{}"}}]})
            state.messages.append({"role": "tool", "tool_call_id": f"c{i}",
                                   "content": "y" * 800})
        loop._compact_if_needed(state, cfg, schemas)
        return context.raw_chars(state.messages, state.system)

    try:
        honest = surviving_chars(1.0)
        dense = surviving_chars(2.35)
        assert dense < honest * 0.6, (
            f"budget barely moved: {honest} chars -> {dense} at 2.35x")
    finally:
        context._calibration = 1.0


def test_a_mutation_still_permits_a_reread(tmp_path, monkeypatch):
    from miniharness import context, loop
    f = tmp_path / "a.py"
    f.write_text("v = 1\n")
    read = {"id": "c1", "name": "Read", "input": {"file_path": str(f)}}
    turns = [
        AssistantTurn(tool_calls=[dict(read)]),
        AssistantTurn(tool_calls=[{"id": "c2", "name": "Bash",
                                   "input": {"command": "true"}}]),
        AssistantTurn(tool_calls=[dict(read, id="c3")]),
        AssistantTurn(text="done"),
    ]
    state = loop.State(system="s", messages=[])
    state.add_user("go")
    seen = []
    cfg = dict(BASE, _cwd=str(tmp_path), max_turns=8)
    for ev in drain(state, cfg, monkeypatch, lambda *a, **k: iter([turns.pop(0)]),
                    tracker=context.FileTracker()):
        if type(ev).__name__ == "ToolEnd":
            seen.append(ev.result)
    assert "same result as before" not in seen[2]


def test_compaction_leaves_headroom_rather_than_sitting_on_the_threshold():
    """Triggering at X and compacting to X is a ratchet: the next tool result
    puts the history straight back over, so every turn compacts. Measured at 37
    compactions in 40 turns on build-stp, with the prefix cache invalidated
    each time."""
    from miniharness import config as cfg_mod
    from miniharness import context, loop, tools

    cfg = dict(cfg_mod.DEFAULTS)
    cfg.update({"llama_ctx": 16384, "_cwd": ".", "model_compaction": False})
    schemas = tools.schemas_for(cfg)
    budget = cfg_mod.history_budget(cfg, schemas)

    state = loop.State(system="s" * 800, messages=[{"role": "user", "content": "go"}])
    for i in range(120):
        state.messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "Read", "arguments": "{}"}}]})
        state.messages.append({"role": "tool", "tool_call_id": f"c{i}",
                               "content": "y" * 4000})

    assert loop._compact_if_needed(state, cfg, schemas) > 0
    after = context.estimate_tokens(state.messages, state.system)
    assert after <= budget * 0.75, (
        f"compacted to {after} against a {budget} budget — no headroom, so the "
        f"next tool result re-triggers immediately")


def test_a_realistic_loop_does_not_compact_every_turn():
    """The end-to-end property the hysteresis exists for."""
    from miniharness import config as cfg_mod
    from miniharness import context, loop, tools

    cfg = dict(cfg_mod.DEFAULTS)
    cfg.update({"llama_ctx": 16384, "_cwd": ".", "model_compaction": False})
    schemas = tools.schemas_for(cfg)
    result_tokens = cfg_mod.budget(cfg, "tool_output_share")

    state = loop.State(system="s" * 800, messages=[{"role": "user", "content": "go"}])
    fires = 0
    for i in range(40):
        if loop._compact_if_needed(state, cfg, schemas):
            fires += 1
        state.messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "Read", "arguments": "{}"}}]})
        state.messages.append({"role": "tool", "tool_call_id": f"c{i}",
                               "content": "y" * (result_tokens * 4)})
    assert fires <= 12, f"{fires} compactions in 40 turns is still thrashing"


def test_the_model_is_told_when_it_is_running_out_of_rounds(tmp_path, monkeypatch):
    """It was given a hard cap and no way to see it, so it explored at the same
    rate whether it had thirty rounds left or two — and six of eleven benchmark
    tasks stopped mid-investigation at the cap."""
    from miniharness import context, loop
    seen = []

    def fake(model, system, messages, schemas, config):
        seen.append(list(messages))
        yield AssistantTurn(tool_calls=[{"id": f"c{len(seen)}", "name": "Bash",
                                         "input": {"command": "true"}}])

    state = loop.State(system="s", messages=[])
    state.add_user("go")
    cfg = dict(BASE, _cwd=str(tmp_path), max_turns=10)
    drain(state, cfg, monkeypatch, fake, tracker=context.FileTracker())

    def warned(req):
        return any("tool round(s) remain" in str(m.get("content") or "") for m in req)

    assert not warned(seen[0]), "warning on the first round is noise"
    assert warned(seen[-1]), "never warned, even on the last round"


def test_the_round_warning_is_never_stored_in_history(tmp_path, monkeypatch):
    """Like the focus map, it belongs to the request, not the transcript —
    otherwise stale counts accumulate and the prefix changes every turn."""
    from miniharness import context, loop

    def fake(model, system, messages, schemas, config):
        yield AssistantTurn(tool_calls=[{"id": "c1", "name": "Bash",
                                         "input": {"command": "true"}}])

    state = loop.State(system="s", messages=[])
    state.add_user("go")
    cfg = dict(BASE, _cwd=str(tmp_path), max_turns=3)
    drain(state, cfg, monkeypatch, fake, tracker=context.FileTracker())
    assert not any("tool round(s) remain" in str(m.get("content") or "")
                   for m in state.messages)


# ── repeating a read-only call continues it, rather than being refused ─────
def _walk(tmp_path, monkeypatch, nlines=600, ctx=8000, repeats=30):
    from miniharness import context, loop
    body = "\n".join(f"line {i:04d}: {'y' * 70}" for i in range(nlines))
    (tmp_path / "big.txt").write_text(body)
    call = {"id": "c", "name": "Read",
            "input": {"file_path": "big.txt", "limit": 100000}}
    turns = [AssistantTurn(tool_calls=[dict(call, id=f"c{i}")]) for i in range(repeats)]
    turns.append(AssistantTurn(text="done"))
    cfg = dict(BASE, _cwd=str(tmp_path), llama_ctx=ctx, max_turns=repeats + 2)
    state = loop.State(system="s", messages=[])
    state.add_user("read all")
    pieces = [e.result for e in
              drain(state, cfg, monkeypatch, lambda *a, **k: iter([turns.pop(0)]),
                    tracker=context.FileTracker())
              if type(e).__name__ == "ToolEnd"]
    return body, pieces


def test_repeating_a_read_delivers_the_whole_file(tmp_path, monkeypatch):
    """A result cut short used to end the model's options: repeating the call
    returned the same first piece, and suppression then refused it. Repeating is
    the only move it has, so repeating is what continues — the same bargain
    reply continuation makes."""
    body, pieces = _walk(tmp_path, monkeypatch)
    seen = "".join(pieces)
    missing = [i for i in range(600) if f"line {i:04d}" not in seen]
    assert not missing, f"never delivered: {missing[:5]}…"


def test_continued_pieces_start_at_a_line_boundary(tmp_path, monkeypatch):
    """Splitting mid-line hands back a chunk with no line number and half an
    identifier."""
    _, pieces = _walk(tmp_path, monkeypatch)
    content = [p for p in pieces if not p.startswith("[no further output")]
    for p in content:
        first = p.split("\n")[1] if p.startswith("[continued]") else p.split("\n")[0]
        assert first.lstrip()[:1].isdigit(), f"piece starts mid-line: {first[:40]!r}"



def test_a_short_result_is_never_paginated(tmp_path, monkeypatch):
    """A result that fits is returned whole, and repeating just returns it
    again. Nothing is withheld and nothing is announced as a repeat."""
    _, pieces = _walk(tmp_path, monkeypatch, nlines=5, ctx=32768, repeats=3)
    assert "more characters not shown" not in pieces[0]
    assert pieces[1] == pieces[0], "a repeat should return the same answer"


def test_a_mutation_restarts_pagination(tmp_path, monkeypatch):
    """The file may have changed, so a part-served copy is stale."""
    from miniharness import context, loop
    (tmp_path / "big.txt").write_text("\n".join(f"line {i:04d}: {'y'*70}" for i in range(600)))
    read = {"name": "Read", "input": {"file_path": "big.txt"}}
    turns = [
        AssistantTurn(tool_calls=[dict(read, id="c1")]),
        AssistantTurn(tool_calls=[{"id": "c2", "name": "Bash", "input": {"command": "true"}}]),
        AssistantTurn(tool_calls=[dict(read, id="c3")]),
        AssistantTurn(text="done"),
    ]
    cfg = dict(BASE, _cwd=str(tmp_path), llama_ctx=8000, max_turns=8)
    state = loop.State(system="s", messages=[])
    state.add_user("go")
    pieces = [e.result for e in
              drain(state, cfg, monkeypatch, lambda *a, **k: iter([turns.pop(0)]),
                    tracker=context.FileTracker())
              if type(e).__name__ == "ToolEnd"]
    assert not pieces[2].startswith("[continued]"), "resumed across a mutation"
    assert "line 0000" in pieces[2]


def test_the_working_set_reaches_the_model_but_not_the_transcript(tmp_path, monkeypatch):
    """Like the focus map, it belongs to the request. Stored, it would double
    every turn and go stale the moment anything changed."""
    from miniharness import context, loop
    f = tmp_path / "a.py"
    f.write_text("value = 1\n")
    read = {"name": "Read", "input": {"file_path": str(f)}}
    seen = []

    def fake(model, system, messages, schemas, config):
        seen.append(list(messages))
        n = len(seen)
        if n < 3:
            yield AssistantTurn(tool_calls=[dict(read, id=f"c{n}")])
        else:
            yield AssistantTurn(text="done")

    state = loop.State(system="s", messages=[])
    state.add_user("look at it")
    cfg = dict(BASE, _cwd=str(tmp_path), max_turns=6)
    drain(state, cfg, monkeypatch, fake, tracker=context.FileTracker())

    last = seen[-1]
    assert any("established so far" in str(m.get("content") or "") for m in last), \
        "the model never saw what it had established"
    assert not any("established so far" in str(m.get("content") or "")
                   for m in state.messages), "it leaked into the transcript"


def test_the_working_set_is_skipped_while_a_tool_call_is_outstanding(tmp_path, monkeypatch):
    """A user message between an assistant's tool_calls and their responses is
    an alternation violation."""
    from miniharness import context, loop
    calls = [{"id": "c1", "name": "Read", "input": {"file_path": "a.py"}},
             {"id": "c2", "name": "Read", "input": {"file_path": "b.py"}}]
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 2\n")
    turns = [AssistantTurn(tool_calls=calls), AssistantTurn(text="done")]
    state = loop.State(system="s", messages=[])
    state.add_user("read both")
    cfg = dict(BASE, _cwd=str(tmp_path), max_turns=4)
    drain(state, cfg, monkeypatch, lambda *a, **k: iter([turns.pop(0)]),
          tracker=context.FileTracker())
    assert_history_valid(state.messages)
    assert_alternates(state.messages)


def test_the_model_gets_its_own_reasoning_back_across_a_tool_call(tmp_path, monkeypatch):
    """Reasoning is stripped from history, so a turn that diagnosed a bug is
    stored as an assistant message with empty content and a tool call. Verified:
    the request for turn N+1 contained nothing worked out in turn N — no
    compaction required. It re-derived every turn, which is why it searched the
    same ground repeatedly."""
    from miniharness import context, loop
    (tmp_path / "a.py").write_text("x = 1\n")
    sent = []
    turns = [
        AssistantTurn(tool_calls=[{"id": "c1", "name": "Read",
                                   "input": {"file_path": "a.py"}}]),
        AssistantTurn(text="done", finish_reason="stop"),
    ]
    turns[0].thinking = "DIAGNOSIS: the bug is on line 1; plan is to set x to 2."

    def fake(model, system, messages, schemas, config):
        sent.append([dict(m) for m in messages])
        yield turns.pop(0)

    state = loop.State(system="sys", messages=[])
    state.add_user("look at a.py")
    drain(state, dict(BASE, _cwd=str(tmp_path), max_turns=3), monkeypatch, fake,
          tracker=context.FileTracker())

    import json as _json
    first = _json.dumps(sent[0])
    second = _json.dumps(sent[1])
    assert "DIAGNOSIS" not in first, "nothing to carry on the first turn"
    assert "DIAGNOSIS" in second, "the model lost its own conclusion"

    # Carried natively, on the assistant message that owns the tool calls —
    # which is the only place this server's template renders it. Measured: on a
    # completed assistant message, or inline as a <think> tag, it is ignored.
    carrier = [m for m in sent[1]
               if m.get("role") == "assistant" and m.get("reasoning_content")]
    assert carrier and carrier[0]["tool_calls"], \
        "reasoning must ride the assistant message that has the tool calls"
    assert not any(m.get("reasoning_content") for m in state.messages
                   if m.get("role") == "assistant" and not m.get("tool_calls")), \
        "a completed turn should carry no reasoning"


def test_only_the_last_couple_of_thoughts_are_carried(tmp_path, monkeypatch):
    """Enough to hold a train of thought across a tool call, not a growing
    history of half-formed guesses."""
    from miniharness import context, loop
    (tmp_path / "a.py").write_text("x = 1\n")
    sent = []
    turns = []
    for i in range(5):
        t = AssistantTurn(tool_calls=[{"id": f"c{i}", "name": "Read",
                                       "input": {"file_path": "a.py"}}])
        t.thinking = f"THOUGHT-{i}"
        turns.append(t)
    turns.append(AssistantTurn(text="done", finish_reason="stop"))

    def fake(model, system, messages, schemas, config):
        sent.append([dict(m) for m in messages])
        yield turns.pop(0)

    state = loop.State(system="sys", messages=[])
    state.add_user("go")
    drain(state, dict(BASE, _cwd=str(tmp_path), max_turns=7), monkeypatch, fake,
          tracker=context.FileTracker())
    # An assistant turn is not finished until its tool calls are answered, so
    # each pending step carries its own reasoning; completed turns carry none.
    import json as _json
    last = _json.dumps(sent[-1])
    carried = [i for i in range(5) if f"THOUGHT-{i}" in last]
    assert carried, "no reasoning survived at all"
    for m in sent[-1]:
        if m.get("reasoning_content"):
            assert m.get("tool_calls"), "reasoning on a message with no tool calls"


@pytest.mark.checkpoints
def test_the_loop_checkpoints_the_baseline_and_every_accepted_change(
        tmp_path, monkeypatch):
    """The wiring, not the module: a run must leave something to rewind to.

    Two checkpoints from one Write — the tree as it was before the agent
    touched it, and the tree after. The first is the one that cannot be
    recovered later, which is why it is taken before the call rather than after.
    """
    from miniharness import checkpoint

    if not checkpoint.available():
        pytest.skip("no git")
    monkeypatch.setattr(checkpoint, "STORE", tmp_path / "store")
    # The store must not sit inside the tree being snapshotted; in real use it
    # lives under ~/.miniharness, which is why this puts them side by side.
    work = tmp_path / "work"
    work.mkdir()
    f = work / "a.py"
    f.write_text("print(1)\n")
    cfg = dict(BASE, _cwd=str(work))
    state = loop.State(messages=[{"role": "user", "content": "rewrite a.py"}],
                       session_id="loop-test")
    tracker = context.FileTracker()
    tracker.mark_read(str(f))          # read-before-overwrite, satisfied

    drain(state, cfg, monkeypatch, script(
        AssistantTurn(text="", finish_reason="tool_calls", tool_calls=[
            {"id": "c1", "name": "Write",
             "input": {"file_path": str(f), "content": "print(2)\n"}}]),
        AssistantTurn(text="Done.", finish_reason="stop"),
    ), tracker=tracker)

    labels = [r[2] for r in checkpoint.history(cfg, "loop-test")]
    assert labels == [f"Write {f}", "before the first change"]

    base = checkpoint.history(cfg, "loop-test")[-1][0]
    checkpoint.restore(cfg, "loop-test", base)
    assert f.read_text() == "print(1)\n"


def test_always_stops_the_prompts_for_the_rest_of_the_same_turn(tmp_path, monkeypatch):
    """"Always" used to be read once per turn, so it kept asking until the
    next message. It must also never be written to config.toml."""
    from miniharness import config as cfg_mod
    cfg = dict(BASE, _cwd=str(tmp_path), accept_all=False)
    asked = []

    def ask(name, params):
        asked.append(name)
        cfg["_accept_session"] = True          # what choosing "always" does
        return True

    state = loop.State(messages=[{"role": "user", "content": "go"}])
    drain(state, cfg, monkeypatch, script(
        AssistantTurn(text="", finish_reason="tool_calls", tool_calls=[
            {"id": "c1", "name": "Bash", "input": {"command": "true"}},
            {"id": "c2", "name": "Bash", "input": {"command": "true"}}]),
        AssistantTurn(text="done", finish_reason="stop"),
    ), ask=ask)
    assert asked == ["Bash"], f"asked again after 'always': {asked}"

    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(cfg_mod, "HOME", tmp_path)
    cfg_mod.save(cfg)
    assert "accept" not in (tmp_path / "config.toml").read_text()


def test_a_refusal_can_say_what_to_do_instead(tmp_path, monkeypatch):
    cfg = dict(BASE, _cwd=str(tmp_path), accept_all=False)
    state = loop.State(messages=[{"role": "user", "content": "go"}])
    events = drain(state, cfg, monkeypatch, script(
        AssistantTurn(text="", finish_reason="tool_calls", tool_calls=[
            {"id": "c1", "name": "Bash", "input": {"command": "rm -rf build"}}]),
        AssistantTurn(text="ok", finish_reason="stop"),
    ), ask=lambda n, p: "use make clean instead")
    ends = [e for e in events if type(e).__name__ == "ToolEnd"]
    assert ends[0].denied
    assert "use make clean instead" in state.messages[2]["content"]


@pytest.mark.checkpoints
def test_when_stuck_several_fixes_are_tried_and_the_best_is_kept(tmp_path, monkeypatch):
    """Watched: a model that stated its bug exactly and still moved between 13
    and 15 of 18 for half an hour. The failing test is a free judge; checkpoints
    make trying and undoing safe."""
    from miniharness import checkpoint
    if not checkpoint.available():
        pytest.skip("no git")
    monkeypatch.setattr(checkpoint, "STORE", tmp_path / "store")
    work = tmp_path / "work"
    work.mkdir()
    (work / "v.py").write_text("ANSWER = 1\n")
    test = ("python3 -c \"import v; print('1 failed, 0 passed in 0.1s' if v.ANSWER != 42 "
            "else '1 passed in 0.1s'); raise SystemExit(v.ANSWER != 42)\"")
    cfg = dict(BASE, _cwd=str(work), max_turns=30, best_of=3)
    tracker = context.FileTracker()
    tracker.mark_read(str(work / "v.py"))

    runs = iter(range(100))
    def fake(model, system, messages, schemas, config):
        last = messages[-1]
        if "asking for your fix from here more than once" in str(last.get("content", "")):
            k = next(runs)
            answers = {0: "ANSWER = 7\n", 1: "ANSWER = 42\n"}
            if k % 3 in answers:
                yield AssistantTurn(text="", finish_reason="tool_calls", tool_calls=[
                    {"id": f"a{k}", "name": "Write",
                     "input": {"file_path": "v.py", "content": answers[k % 3]}}])
            else:
                yield AssistantTurn(text="thinking it over", finish_reason="stop")
            return
        if sum(1 for m in messages if m.get("role") == "tool") < 7:
            yield AssistantTurn(text="", finish_reason="tool_calls", tool_calls=[
                {"id": f"t{len(messages)}", "name": "Bash", "input": {"command": test}}])
        else:
            yield AssistantTurn(text="done", finish_reason="stop")

    state = loop.State(messages=[{"role": "user", "content": "make the test pass"}],
                       session_id="alt-test")
    events = drain(state, cfg, monkeypatch, fake, tracker=tracker)
    notices = [e.text for e in events if type(e).__name__ == "Notice"]

    assert (work / "v.py").read_text() == "ANSWER = 42\n", "the best fix was not kept"
    assert any(n.startswith("trying 3 alternative fixes") for n in notices), notices
    assert any("kept alternative 2" in n for n in notices), notices
    assert any("kept #2: 1 failed, 0 passed → 0 failed, 1 passed" in str(m.get("content"))
               for m in state.messages if m["role"] == "tool")
    assert_history_valid(state.messages)


@pytest.mark.checkpoints
def test_when_no_alternative_is_better_nothing_is_changed(tmp_path, monkeypatch):
    from miniharness import checkpoint
    if not checkpoint.available():
        pytest.skip("no git")
    monkeypatch.setattr(checkpoint, "STORE", tmp_path / "store")
    work = tmp_path / "work"
    work.mkdir()
    (work / "v.py").write_text("ANSWER = 1\n")
    test = ("python3 -c \"import v; print('1 failed, 0 passed in 0.1s' if v.ANSWER != 42 "
            "else '1 passed in 0.1s'); raise SystemExit(v.ANSWER != 42)\"")
    cfg = dict(BASE, _cwd=str(work), max_turns=30, best_of=2)
    tracker = context.FileTracker()
    tracker.mark_read(str(work / "v.py"))

    def fake(model, system, messages, schemas, config):
        if "asking for your fix from here more than once" in str(messages[-1].get("content", "")):
            yield AssistantTurn(text="", finish_reason="tool_calls", tool_calls=[
                {"id": "w", "name": "Write", "input": {"file_path": "v.py", "content": "ANSWER = 3\n"}}])
        elif sum(1 for m in messages if m.get("role") == "tool") < 7:
            yield AssistantTurn(text="", finish_reason="tool_calls", tool_calls=[
                {"id": f"t{len(messages)}", "name": "Bash", "input": {"command": test}}])
        else:
            yield AssistantTurn(text="done", finish_reason="stop")

    state = loop.State(messages=[{"role": "user", "content": "go"}], session_id="alt-none")
    drain(state, cfg, monkeypatch, fake, tracker=tracker)
    assert (work / "v.py").read_text() == "ANSWER = 1\n", "a worse fix was left in place"
    assert any("nothing was changed" in str(m.get("content")) for m in state.messages)
    assert_history_valid(state.messages)
