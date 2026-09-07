"""Provider streaming against a real HTTP server.

The scripted loop tests bypass ``provider.stream`` entirely, which leaves the
SSE parser — delta accumulation, index-keyed tool-call assembly, finish_reason
— untested. This spins up a real socket server and replays recorded wire
formats through it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from miniharness import config as cfg_mod
from miniharness.provider import AssistantTurn, TextChunk, stream


def sse(*events) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return (body + "data: [DONE]\n\n").encode()


def delta(content=None, tool_calls=None, finish=None):
    d = {}
    if content is not None:
        d["content"] = content
    if tool_calls is not None:
        d["tool_calls"] = tool_calls
    return {"choices": [{"delta": d, "finish_reason": finish}]}


@pytest.fixture
def server():
    """A one-shot OpenAI-compatible endpoint. Set ``server.body`` per test."""
    state = {"body": b"", "status": 200, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            state["requests"].append(json.loads(self.rfile.read(n) or b"{}"))
            self.send_response(state["status"])
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(state["body"])

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["port"] = httpd.server_address[1]
    yield state
    httpd.shutdown()


def run(server, schemas=()):
    cfg = dict(cfg_mod.DEFAULTS)
    cfg["llama_port"] = server["port"]
    cfg["model"] = "local"
    return list(stream("local", "sys", [{"role": "user", "content": "hi"}],
                       list(schemas), cfg))


def test_text_streams_in_chunks_then_yields_one_turn(server):
    server["body"] = sse(delta("Hello"), delta(", world"), delta(finish="stop"))
    events = run(server)
    chunks = [e.text for e in events if isinstance(e, TextChunk)]
    turns = [e for e in events if isinstance(e, AssistantTurn)]
    assert chunks == ["Hello", ", world"]
    assert len(turns) == 1
    assert turns[0].text == "Hello, world"
    assert turns[0].finish_reason == "stop"
    assert not turns[0].truncated


def test_tool_call_arguments_are_reassembled_from_deltas(server):
    server["body"] = sse(
        delta(tool_calls=[{"index": 0, "id": "c1",
                           "function": {"name": "Read", "arguments": '{"file_'}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": 'path": "a.py"}'}}]),
        delta(finish="tool_calls"),
    )
    turn = run(server)[-1]
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0]["name"] == "Read"
    assert turn.tool_calls[0]["input"] == {"file_path": "a.py"}


def test_parallel_tool_calls_keep_their_indices_apart(server):
    server["body"] = sse(
        delta(tool_calls=[{"index": 0, "id": "a",
                           "function": {"name": "Read", "arguments": '{"file_path":"a"}'}}]),
        delta(tool_calls=[{"index": 1, "id": "b",
                           "function": {"name": "Glob", "arguments": '{"pattern":"*.py"}'}}]),
        delta(finish="tool_calls"),
    )
    turn = run(server)[-1]
    assert [tc["name"] for tc in turn.tool_calls] == ["Read", "Glob"]
    assert turn.tool_calls[1]["input"] == {"pattern": "*.py"}


def test_truncated_arguments_are_marked_not_guessed(server):
    """Malformed JSON must surface as _raw so the loop can drop the call."""
    server["body"] = sse(
        delta(tool_calls=[{"index": 0, "id": "c1",
                           "function": {"name": "Edit", "arguments": '{"file_pa'}}]),
        delta(finish="length"),
    )
    turn = run(server)[-1]
    assert turn.truncated
    assert "_raw" in turn.tool_calls[0]["input"]


def test_text_emitted_tool_calls_are_recovered(server):
    """The local-model case: a call written into the body, not the field."""
    server["body"] = sse(
        delta('<tool_call>{"name":"Read","arguments":{"file_path":"a.py"}}</tool_call>'),
        delta(finish="stop"),
    )
    schemas = [{"name": "Read", "description": "d", "input_schema": {"type": "object"}}]
    turn = run(server, schemas)[-1]
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0]["name"] == "Read"
    assert "tool_call" not in turn.text


def test_recovery_is_off_when_configured_off(server):
    server["body"] = sse(delta('{"name":"Read","arguments":{"file_path":"a"}}'),
                         delta(finish="stop"))
    cfg = dict(cfg_mod.DEFAULTS)
    cfg["llama_port"] = server["port"]
    cfg["recover_text_tool_calls"] = False
    schemas = [{"name": "Read", "description": "d", "input_schema": {"type": "object"}}]
    events = list(stream("local", "s", [{"role": "user", "content": "x"}], schemas, cfg))
    assert events[-1].tool_calls == []


def test_the_system_message_leads_the_payload(server):
    """Prefix stability: the system message must be first, every time."""
    server["body"] = sse(delta("ok"), delta(finish="stop"))
    run(server)
    sent = server["requests"][-1]
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][0]["content"] == "sys"


def test_tools_are_sent_in_declared_order(server):
    server["body"] = sse(delta("ok"), delta(finish="stop"))
    from miniharness import tools
    run(server, tools.schemas_for({}))
    sent = server["requests"][-1]
    assert [t["function"]["name"] for t in sent["tools"]] == \
        [s["name"] for s in tools.SCHEMAS]


def test_http_errors_surface_with_the_body(server):
    server["status"] = 500
    server["body"] = b"upstream exploded"
    with pytest.raises(RuntimeError, match="500"):
        run(server)


def test_malformed_sse_lines_are_skipped_not_fatal(server):
    server["body"] = (b"data: {not json}\n\n"
                      b": a comment line\n\n"
                      + sse(delta("fine"), delta(finish="stop")))
    turn = run(server)[-1]
    assert turn.text == "fine"


# ── continuation: a generation cap must not cost the model anything ─────────
def _turn(text, finish, calls=None):
    return AssistantTurn(text=text, finish_reason=finish, tool_calls=calls or [])


def _scripted(monkeypatch, turns):
    """Replace the single-request stream with a scripted sequence."""
    from miniharness import provider
    seq = list(turns)
    seen = []

    def fake(model, system, messages, schemas, config):
        seen.append(messages)
        t = seq.pop(0)
        if t.text:
            yield TextChunk(t.text)
        yield t
    monkeypatch.setattr(provider, "stream", fake)
    return seen


def test_a_capped_reply_is_continued_and_merged(monkeypatch):
    """A cap is the harness's constraint, not the model's. Writing past it used
    to fragment the reply across two assistant messages with a user message
    wedged between; now it is one clean turn."""
    from miniharness import provider
    _scripted(monkeypatch, [_turn("first half ", "length"), _turn("second half", "stop")])
    out = list(provider.stream_complete("local", "sys", [{"role": "user", "content": "go"}],
                                        [], {"max_continuations": 8}))
    final = out[-1]
    assert final.text == "first half second half"
    assert not final.truncated
    assert final.to_message()["content"] == "first half second half"


def test_continuation_resumes_rather_than_restarting(monkeypatch):
    seen = _scripted(monkeypatch, [_turn("abc", "length"), _turn("def", "stop")])
    from miniharness import provider
    list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                  [], {"max_continuations": 8}))
    resumed = seen[1]
    assert resumed[-2]["role"] == "assistant" and resumed[-2]["content"] == "abc"
    assert "Continue" in resumed[-1]["content"]


def test_the_model_is_never_told_to_write_less(monkeypatch):
    """Running past the cap is the harness's problem, not the model's. What we
    send back must ask it to resume, never to shorten."""
    seen = _scripted(monkeypatch, [_turn("abc", "length"), _turn("def", "stop")])
    from miniharness import provider
    list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                  [], {"max_continuations": 8}))
    nudge = seen[1][-1]["content"].lower()
    assert "concise" not in nudge and "shorter" not in nudge and "brief" not in nudge
    assert "do not repeat" in nudge


def test_many_continuations_all_merge(monkeypatch):
    from miniharness import provider
    _scripted(monkeypatch, [_turn(f"p{i} ", "length") for i in range(4)]
              + [_turn("end", "stop")])
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], {"max_continuations": 8}))
    assert out[-1].text == "p0 p1 p2 p3 end"


def test_a_tool_call_stops_continuation(monkeypatch):
    """A tool call is actionable immediately; only prose needs continuing."""
    from miniharness import provider
    _scripted(monkeypatch, [_turn("thinking", "length",
                                  [{"id": "c1", "name": "Read", "input": {"file_path": "a"}}])])
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], {"max_continuations": 8}))
    assert out[-1].tool_calls and out[-1].tool_calls[0]["name"] == "Read"


def test_continuation_is_bounded(monkeypatch):
    """A model that never stops must not loop forever."""
    from miniharness import provider
    _scripted(monkeypatch, [_turn("x", "length") for _ in range(20)])
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], {"max_continuations": 3}))
    assert out[-1].truncated and out[-1].text == "xxxx"


def test_a_cap_spent_entirely_on_thinking_does_not_scaffold_an_empty_turn(monkeypatch):
    """A thinking model can burn the whole cap inside <think> and emit no
    visible text. Resuming from an empty assistant message breaks user/assistant
    alternation — exactly what the truncation fix exists to prevent."""
    from miniharness import provider
    seen = _scripted(monkeypatch, [_turn("", "length"), _turn("the answer", "stop")])
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], {"max_continuations": 8}))
    assert not any(m.get("role") == "assistant" and not (m.get("content") or "").strip()
                   for req in seen for m in req), "empty assistant message scaffolded"
    assert seen[1] == seen[0], "should re-issue the original request, not scaffold"
    assert out[-1].text == "the answer"


def test_whitespace_only_output_is_treated_as_nothing_written(monkeypatch):
    from miniharness import provider
    seen = _scripted(monkeypatch, [_turn("   \n", "length"), _turn("real text", "stop")])
    list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                  [], {"max_continuations": 8}))
    assert seen[1] == seen[0]


def test_continuation_stops_before_it_overflows_the_window(monkeypatch):
    """Each round resends everything written so far, and the loop compacts
    before this call rather than during it. Without a window bound a long reply
    would grow the request past the context window and 400 mid-turn — turning a
    long answer into a failed turn."""
    from miniharness import provider
    # 8k window, 25% reply share -> ~2k reply; each round writes ~1k tokens.
    cfg = {"llama_ctx": 8192, "reply_share": 0.25, "compact_at": 0.85,
           "max_continuations": 50}
    _scripted(monkeypatch, [_turn("x" * 4000, "length") for _ in range(50)])
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], cfg))
    final = out[-1]
    used = len(final.text) // 4
    assert used < 8192, f"accumulated {used} tokens into an 8192 window"
    assert final.truncated, "stopping for want of room must report as truncated"


def test_the_window_bound_does_not_fire_early(monkeypatch):
    """A short reply in a large window must still continue normally."""
    from miniharness import provider
    cfg = {"llama_ctx": 131072, "reply_share": 0.25, "compact_at": 0.85,
           "max_continuations": 8}
    _scripted(monkeypatch, [_turn("a", "length"), _turn("b", "length"), _turn("c", "stop")])
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], cfg))
    assert out[-1].text == "abc" and not out[-1].truncated


def test_the_two_continuation_budgets_are_separate():
    """stream_complete and the loop each retry a truncated reply, at different
    layers. Sharing one key would silently double the cap and hide which layer
    was doing the work."""
    import inspect
    import re

    from miniharness.config import DEFAULTS
    assert DEFAULTS["max_continuations"] != DEFAULTS["max_truncation_resumes"]
    from miniharness import loop, provider
    # Assert on the actual config lookups, not the prose: the comment in
    # _handle_truncation names the other key precisely to explain the split.
    def keys_read(fn):
        return set(re.findall(r'config\.get\(\s*"(\w+)"', inspect.getsource(fn)))

    assert "max_truncation_resumes" in keys_read(loop._handle_truncation)
    assert "max_continuations" not in keys_read(loop._handle_truncation)
    assert "max_continuations" in keys_read(provider.stream_complete)
    assert "max_truncation_resumes" not in keys_read(provider.stream_complete)


def test_a_tool_call_split_by_the_cap_is_recovered_from_the_merged_text(monkeypatch):
    """Small models write tool calls into the message body. Recovery runs per
    round, so a call cut in half by the cap parses in neither half — but the
    merged text holds it intact. Without a re-scan the call silently becomes
    prose: the model says it will run a command and nothing runs."""
    from miniharness import provider
    half1 = 'I will list the files.\n```json\n{"name": "Bash", "arguments": {"comm'
    half2 = 'and": "ls -la /app"}}\n```\n'
    _scripted(monkeypatch, [_turn(half1, "length"), _turn(half2, "stop")])
    schemas = [{"name": "Bash", "description": "run", "input_schema": {}}]
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        schemas, {"max_continuations": 8,
                                                  "recover_text_tool_calls": True}))
    final = out[-1]
    assert final.tool_calls, "tool call split across the seam was lost"
    assert final.tool_calls[0]["name"] == "Bash"
    assert final.tool_calls[0]["input"]["command"] == "ls -la /app"


def test_recovery_across_the_seam_does_not_disturb_a_normal_reply(monkeypatch):
    from miniharness import provider
    _scripted(monkeypatch, [_turn("just ", "length"), _turn("prose here", "stop")])
    schemas = [{"name": "Bash", "description": "run", "input_schema": {}}]
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        schemas, {"max_continuations": 8}))
    assert not out[-1].tool_calls and out[-1].text == "just prose here"


def test_a_network_failure_mid_continuation_keeps_what_was_written(monkeypatch):
    """Discarding four rounds of a long reply because round five could not
    connect penalises the model for a network fault."""
    from miniharness import provider
    seq = [_turn("part one. ", "length"), _turn("part two. ", "length")]

    def fake(model, system, messages, schemas, config):
        if not seq:
            raise RuntimeError("could not reach server: ConnectionError")
        t = seq.pop(0)
        yield TextChunk(t.text)
        yield t
    monkeypatch.setattr(provider, "stream", fake)
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], {"max_continuations": 8}))
    final = out[-1]
    assert final.text == "part one. part two. "
    assert final.truncated and final.finish_reason == "network"


def test_a_network_failure_on_the_very_first_round_still_raises(monkeypatch):
    """With nothing written there is nothing to salvage, and swallowing the
    error would turn an unreachable server into a silent empty turn."""
    from miniharness import provider

    def fake(*a, **kw):
        raise RuntimeError("could not reach server")
        yield  # pragma: no cover
    monkeypatch.setattr(provider, "stream", fake)
    with pytest.raises(RuntimeError):
        list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                      [], {"max_continuations": 8}))


def test_reasoning_is_carried_across_the_seam_not_thrown_away(monkeypatch):
    """A cap landing inside <think> produces no visible text but a great deal
    of work. Re-issuing the original request makes the model reason from
    scratch: measured at 12,907 tokens of thinking discarded over three
    identical retries, 393 s for one turn."""
    from miniharness import provider
    t1 = _turn("", "length"); t1.thinking = "step one: consider the cases"
    t2 = _turn("the answer is 42", "stop")
    seen = _scripted(monkeypatch, [t1, t2])
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], {"max_continuations": 8}))
    resumed = seen[1]
    assert resumed != seen[0], "re-issued the same request, discarding the reasoning"
    assert resumed[-2]["role"] == "assistant"
    assert "step one" in resumed[-2]["content"]
    assert "cut off by a length limit" in resumed[-1]["content"]
    assert out[-1].text == "the answer is 42"


def test_carried_reasoning_never_reaches_stored_history(monkeypatch):
    """Thinking is fed back only to finish the turn. It must not end up in the
    message the loop appends, or it costs context on every later request."""
    from miniharness import provider
    t1 = _turn("", "length"); t1.thinking = "SECRET REASONING"
    t2 = _turn("done", "stop")
    _scripted(monkeypatch, [t1, t2])
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], {"max_continuations": 8}))
    assert "SECRET REASONING" not in json.dumps(out[-1].to_message())


def test_nothing_at_all_still_re_issues_unchanged(monkeypatch):
    """No text and no reasoning: there is nothing to resume from, and
    scaffolding an empty assistant message would break alternation."""
    from miniharness import provider
    seen = _scripted(monkeypatch, [_turn("", "length"), _turn("ok", "stop")])
    list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                  [], {"max_continuations": 8}))
    assert seen[1] == seen[0]


def test_a_long_chain_of_thought_cannot_overflow_the_window(monkeypatch):
    """When the cap lands inside <think> it is the reasoning that is resent
    each round, so measuring only the visible text would let it overflow
    unseen."""
    from miniharness import provider
    seq = []
    for _ in range(50):
        t = _turn("", "length"); t.thinking = "r" * 4000
        seq.append(t)
    _scripted(monkeypatch, seq)
    cfg = {"llama_ctx": 8192, "reply_share": 0.25, "compact_at": 0.85,
           "repo_map_share": 0.05, "max_continuations": 50}
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], cfg))
    assert len(out[-1].thinking) // 4 < 8192, "reasoning grew past the window"


def test_when_reasoning_fills_the_window_the_model_is_asked_to_land(monkeypatch):
    """Ending the turn empty wastes everything: the loop retries and pays the
    same cost to reach the same place. Measured at 11,704 tokens of thinking
    and no tool call, twice over."""
    from miniharness import provider
    # ~5k thinking tokens per round against a 13,230-token budget: the bound
    # trips after the third, and the fourth request is the landing round.
    # (The budget grew when the reply reservation was removed — see
    # config.history_budget — so it now takes one more round to fill.)
    seq = []
    for _ in range(3):
        t = _turn("", "length"); t.thinking = "r" * 20000
        seq.append(t)
    seq.append(_turn("", "stop", [{"id": "c1", "name": "Bash", "input": {"command": "ls"}}]))
    seen = _scripted(monkeypatch, seq)
    cfg = {"llama_ctx": 16384, "reply_share": 0.25, "compact_at": 0.85,
           "repo_map_share": 0.05, "max_continuations": 50}
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], cfg))
    assert out[-1].tool_calls, "turn ended with nothing after filling the window"
    assert "Stop deliberating" in seen[-1][-1]["content"]
    # the reasoning was handed back, not discarded (possibly trimmed to fit)
    assert seen[-1][-2]["role"] == "assistant"
    assert seen[-1][-2]["content"].rstrip().endswith("r")


def test_the_landing_round_keeps_its_text(monkeypatch):
    from miniharness import provider
    seq = []
    for _ in range(2):
        t = _turn("", "length"); t.thinking = "r" * 20000
        seq.append(t)
    seq.append(_turn("FINAL ANSWER", "stop"))
    _scripted(monkeypatch, seq)
    cfg = {"llama_ctx": 16384, "reply_share": 0.25, "compact_at": 0.85,
           "repo_map_share": 0.05, "max_continuations": 50}
    out = list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                        [], cfg))
    assert out[-1].text == "FINAL ANSWER"


def test_no_landing_round_when_there_is_visible_text(monkeypatch):
    """A partial answer is resumable normally; landing is only for the case
    where the whole window went into reasoning."""
    from miniharness import provider
    seq = []
    for _ in range(4):
        t = _turn("some answer text ", "length"); t.thinking = "r" * 20000
        seq.append(t)
    seen = _scripted(monkeypatch, seq)
    cfg = {"llama_ctx": 16384, "reply_share": 0.25, "compact_at": 0.85,
           "repo_map_share": 0.05, "max_continuations": 50}
    list(provider.stream_complete("local", "s", [{"role": "user", "content": "go"}],
                                  [], cfg))
    assert all("Stop deliberating" not in m["content"]
               for req in seen for m in req if m["role"] == "user")


def test_the_landing_round_fits_the_window():
    """It is called *because* the window bound tripped, so history + reasoning
    is over budget by construction. Sending it whole would 400 — the landing
    round becoming the failure it exists to avoid."""
    import json as _json

    from miniharness import config as cfgmod
    from miniharness import context, provider, tools

    window = 16384
    cfg = dict(cfgmod.DEFAULTS)
    cfg.update({"llama_ctx": window, "_cwd": ".", "model": "local"})
    schemas = tools.schemas_for(cfg)
    messages = [{"role": "user", "content": "solve it " * 200}]
    system = "s" * 800
    reasoned = "r" * 200_000          # far larger than the window

    captured = {}

    def fake(model, sys_, msgs, sch, conf):
        captured["req"] = msgs
        yield AssistantTurn(text="done", finish_reason="stop")
    import miniharness.provider as P
    orig, P.stream = P.stream, fake
    try:
        list(provider._conclude("local", system, messages, reasoned, schemas, cfg))
    finally:
        P.stream = orig

    sent = (context.estimate_tokens(captured["req"], system)
            + len(_json.dumps(schemas)) // 4)
    # Nothing is reserved for the reply now; what must hold is that the landing
    # request still leaves the model somewhere to write.
    limit = int(window * float(cfg.get("compact_at", 0.85)))
    assert sent <= limit, f"landing request {sent} leaves no room: > {limit} of {window}"


def test_the_landing_round_keeps_the_end_of_the_reasoning():
    """The end of a chain of thought is where its conclusions are."""
    from miniharness import config as cfgmod
    from miniharness import provider, tools
    cfg = dict(cfgmod.DEFAULTS)
    cfg.update({"llama_ctx": 16384, "_cwd": ".", "model": "local"})
    captured = {}

    def fake(model, sys_, msgs, sch, conf):
        captured["req"] = msgs
        yield AssistantTurn(text="done", finish_reason="stop")
    import miniharness.provider as P
    orig, P.stream = P.stream, fake
    try:
        list(provider._conclude("local", "s", [{"role": "user", "content": "go"}],
                                "START" + "r" * 200_000 + "CONCLUSION",
                                tools.schemas_for(cfg), cfg))
    finally:
        P.stream = orig
    body = captured["req"][-2]["content"]
    assert body.endswith("CONCLUSION")
    assert "trimmed to fit" in body


def test_usage_is_requested_so_calibration_can_engage():
    """An OpenAI-compatible server sends no usage block on a streamed response
    unless asked. Without it the estimator has nothing to calibrate against and
    silently keeps its own bad guess — the failure mode being that everything
    looks fine and the window overflows anyway."""
    import inspect
    from miniharness import provider
    src = inspect.getsource(provider)
    assert '"stream_options"' in src and "include_usage" in src


def test_calibration_runs_on_a_real_streamed_usage_block(monkeypatch):
    from miniharness import context, provider
    context._calibration = 1.0
    body = (
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
        'data: {"choices":[],"usage":{"prompt_tokens":8000,"completion_tokens":2}}\n'
        'data: [DONE]\n'
    )

    class R:
        status_code = 200
        headers = {}
        def iter_lines(self, decode_unicode=False):
            for line in body.splitlines():
                yield line
        def close(self): pass

    monkeypatch.setattr(provider, "_connect", lambda *a, **k: R())
    msgs = [{"role": "user", "content": "x" * 16_000}]   # 4,000 by the heuristic
    try:
        list(provider.stream("local", "", msgs, [], {"llama_host": "h", "llama_port": 1,
                                                     "model": "local"}))
        assert context._calibration == 2.0, context._calibration
    finally:
        context._calibration = 1.0


def test_thinking_can_be_disabled_for_mechanical_work(monkeypatch):
    """Summarising what already happened does not need deliberation, and on this
    model the default spent all 300 tokens inside <think> and returned nothing —
    a whole generation for an empty answer."""
    from miniharness import provider
    seen = {}

    class R:
        status_code = 200
        headers = {}
        def iter_lines(self, decode_unicode=False):
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
            yield "data: [DONE]"
        def close(self): pass

    def fake_connect(url, headers, payload, config):
        seen["payload"] = payload
        return R()

    monkeypatch.setattr(provider, "_connect", fake_connect)
    cfg = {"llama_host": "h", "llama_port": 1, "model": "local", "disable_thinking": True}
    list(provider.stream("local", "s", [{"role": "user", "content": "go"}], [], cfg))
    assert seen["payload"]["chat_template_kwargs"] == {"enable_thinking": False}

    monkeypatch.setattr(provider, "_connect", fake_connect)
    list(provider.stream("local", "s", [{"role": "user", "content": "go"}], [],
                         {"llama_host": "h", "llama_port": 1, "model": "local"}))
    assert "chat_template_kwargs" not in seen["payload"], \
        "the agent's own turns must keep their reasoning"


@pytest.mark.real_summariser
def test_the_summary_call_uses_the_mechanical_settings():
    import inspect
    from miniharness import context
    src = inspect.getsource(context.summarise_span)
    assert "disable_thinking=True" in src
    # Greedy decoding alone loops: 42 identical lines, the whole cap, 101s.
    assert "frequency_penalty=0.4" in src


def test_only_the_mechanical_call_carries_a_frequency_penalty(monkeypatch):
    """Greedy decoding sent the summariser into a loop — 42 identical lines and
    the whole cap spent, 101s of the ~150s a compaction costs. The penalty stops
    that. It must not reach the agent's turns, where repeating an identifier is
    the job rather than a fault."""
    from miniharness import provider
    seen = {}

    class R:
        status_code = 200
        headers = {}
        def iter_lines(self, decode_unicode=False):
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
            yield "data: [DONE]"
        def close(self): pass

    monkeypatch.setattr(provider, "_connect",
                        lambda url, headers, payload, config:
                        (seen.__setitem__("payload", payload), R())[1])

    base = {"llama_host": "h", "llama_port": 1, "model": "local"}
    msgs = [{"role": "user", "content": "go"}]

    list(provider.stream("local", "s", msgs, [], dict(base, frequency_penalty=0.4)))
    assert seen["payload"]["frequency_penalty"] == 0.4

    list(provider.stream("local", "s", msgs, [], base))
    assert "frequency_penalty" not in seen["payload"], \
        "the agent must be free to repeat an identifier"


def test_a_runaway_turn_is_bounded_by_rounds_and_announced(monkeypatch):
    """A turn that keeps hitting the cap must not run for an hour in silence.

    Two separate runs were diagnosed as hangs when the loop was simply
    generating, emitting nothing, because the "continued Nx" notice was only
    reported once the turn ended — and the turn whose cost matters is the one
    that has not ended. So each round announces itself as it starts.

    What bounds the turn is the round cap and the window, not a clock. A
    seconds limit was tried and removed: it made the same model on the same
    task behave differently on faster and slower hardware, which is the harness
    working against the model rather than with it.
    """
    import time
    from miniharness import provider

    clock = {"t": 0.0}
    monkeypatch.setattr(provider._time, "monotonic", lambda: clock["t"])

    def fake_stream(model, system, messages, schemas, config):
        clock["t"] += 100.0                      # each round burns 100 seconds
        yield provider.AssistantTurn(text="more and more text ",
                                     finish_reason="length")

    monkeypatch.setattr(provider, "stream", fake_stream)

    events = list(provider.stream_complete(
        "local", "sys", [{"role": "user", "content": "write forever"}], [],
        {"max_continuations": 8, "llama_ctx": 32768}))

    announced = [e for e in events if isinstance(e, provider.Continuing)]
    assert announced, "a continuing turn said nothing while it ran"
    assert announced[0].round == 1 and announced[0].elapsed > 0

    # Every round says so as it starts, so a long turn is never silent.
    assert [a.round for a in announced] == list(range(1, len(announced) + 1))
    # And it stops: the round cap and the window still bound it. No clock does.
    assert len(announced) <= 8, "the round cap did not hold"

    # Nothing the model wrote is thrown away.
    final = [e for e in events if isinstance(e, provider.AssistantTurn)]
    assert final and "more and more text" in final[-1].text


def test_the_agent_turn_is_not_capped_but_a_mechanical_call_is(monkeypatch):
    """No generation cap on the agent's own turn.

    `reply_share` used to do two jobs: reserve window for the answer and cap
    what the model could generate. Neither survived scrutiny. As a cap it
    bounded nothing in aggregate — 8 continuations of a 32,768-token cap allow
    ~295k tokens in one turn, more than twice a 128k window — while creating
    every seam the continuation machinery then had to repair, each seam costing
    a full resend. Measured at a 655-token cap: all 8 rounds burned and no tool
    call at all in 2 of 3 turns.

    It survives only where a bound is part of the definition of the task: a
    compaction note longer than the span it replaces reclaims nothing.
    """
    from miniharness import provider
    seen = {}

    class R:
        status_code = 200
        headers = {}
        def iter_lines(self, decode_unicode=False):
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
            yield "data: [DONE]"
        def close(self): pass

    monkeypatch.setattr(provider, "_connect",
                        lambda url, headers, payload, config:
                        (seen.__setitem__("payload", payload), R())[1])

    base = {"llama_host": "h", "llama_port": 1, "model": "local", "llama_ctx": 32768}
    msgs = [{"role": "user", "content": "go"}]

    list(provider.stream("local", "s", msgs, [], base))
    assert "max_tokens" not in seen["payload"], \
        "the agent's turn must run until the model is finished"

    list(provider.stream("local", "s", msgs, [], dict(base, reply_share=0.06)))
    assert seen["payload"]["max_tokens"] == 1966

    from miniharness import config as cfg_mod
    assert "reply_share" not in cfg_mod.DEFAULTS, \
        "a default reply_share would silently re-cap every turn"


def test_no_window_is_reserved_for_the_reply():
    """History gets everything that is genuinely free.

    A reservation limits in both directions — it caps the answer at the size
    set aside and shrinks the history whether or not the answer needs it. The
    reply is the work; the history is what yields, after the fact, by
    compaction.
    """
    from miniharness import config as cfg_mod
    cfg = {"llama_ctx": 32768, "compact_at": 0.85, "repo_map_share": 0.05}
    room = 32768 - cfg_mod.budget(cfg, "repo_map_share", floor=0)
    assert cfg_mod.history_budget(cfg, []) == int(room * 0.85)


def test_circling_reasoning_is_ended_with_its_work_kept(monkeypatch):
    """A loop that never converges must end in a decision, not a dead turn.

    Qwen documents this in its own model card: no strong stop-thinking signal,
    so the model "rephrases the same logic and revisits the same conclusions".
    Measured here at 45,101 tokens in one turn, and at 13,000 while the model
    wrote "Actually, I think I've been going in circles."

    The reasoning is handed to the landing round, never discarded — discarding
    it is §2.3, which cost 393s for one turn.
    """
    from miniharness import provider
    from miniharness.provider import ThinkChunk

    circle = ("Let me search for the pattern in the modules.\n"
              "Actually, let me look at the results more carefully.\n"
              "Let me search for the pattern in the modules.\n") * 200

    calls = []

    def fake(model, system, messages, schemas, config):
        calls.append(messages)
        if len(calls) == 1:
            for i in range(0, len(circle), 500):
                yield ThinkChunk(circle[i:i + 500])
            yield AssistantTurn(text="", finish_reason="length")
        else:                      # the landing round
            yield AssistantTurn(text="done", finish_reason="stop", tool_calls=[
                {"id": "c1", "name": "Bash", "input": {"command": "ls"}}])

    monkeypatch.setattr(provider, "stream", fake)
    out = list(provider.stream_complete(
        "local", "sys", [{"role": "user", "content": "go"}], [],
        {"llama_ctx": 32768, "max_continuations": 8}))

    stopped = [e for e in out if isinstance(e, provider.StoppedCircling)]
    assert stopped, "the loop was never interrupted"
    assert "distinct" in stopped[0].reason, stopped[0].reason

    assert len(calls) == 2, "the landing round did not run"
    handed_back = calls[1][-2]["content"]
    assert "Let me search for the pattern" in handed_back, \
        "the reasoning was discarded instead of carried into the landing round"
    assert out[-1].tool_calls, "the turn ended without an action"


def test_a_long_but_progressing_turn_is_left_alone(monkeypatch):
    """The stop must not fire on a model that is simply thinking hard."""
    from miniharness import provider
    from miniharness.provider import ThinkChunk

    varied = "".join(f"Considering case {i}: the index is computed from n-{i}.\n"
                     for i in range(400))

    def fake(model, system, messages, schemas, config):
        for i in range(0, len(varied), 500):
            yield ThinkChunk(varied[i:i + 500])
        yield AssistantTurn(text="ok", finish_reason="stop", tool_calls=[
            {"id": "c1", "name": "Bash", "input": {"command": "ls"}}])

    monkeypatch.setattr(provider, "stream", fake)
    out = list(provider.stream_complete(
        "local", "sys", [{"role": "user", "content": "go"}], [],
        {"llama_ctx": 32768, "max_continuations": 8}))
    assert not [e for e in out if isinstance(e, provider.StoppedCircling)]
    assert out[-1].tool_calls


def test_the_landing_round_is_bounded_too(monkeypatch):
    """The one place an unbounded call is guaranteed to hurt.

    The landing round is reached *because* the model was already circling, so
    leaving it unwatched is the worst case rather than an edge case. Measured:
    a landing round ran to 31,191 tokens over 31 minutes while the round loop's
    own bound of 13,107 sat one function away, unused.

    There is nothing to escalate to, so the partial turn comes back marked
    truncated and the loop's existing handling takes over — with the reasoning
    kept, which is the whole point of landing rather than restarting.
    """
    from miniharness import provider
    from miniharness.provider import ThinkChunk

    circle = ("Let me reconsider the ranking formula once more.\n"
              "Actually, let me approach this differently.\n") * 400

    def fake(model, system, messages, schemas, config):
        for i in range(0, len(circle), 400):
            yield ThinkChunk(circle[i:i + 400])
        raise AssertionError("the landing round was never interrupted")

    monkeypatch.setattr(provider, "stream", fake)
    out = list(provider._conclude("local", "sys",
                                  [{"role": "user", "content": "go"}],
                                  "earlier reasoning", [],
                                  {"llama_ctx": 32768}))
    # The generator's return value is the turn; drain it to get there.
    gen = provider._conclude("local", "sys", [{"role": "user", "content": "go"}],
                             "earlier reasoning", [], {"llama_ctx": 32768})
    turn = None
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        turn = stop.value
    assert turn is not None, "no turn came back from an interrupted landing round"
    assert turn.truncated
    assert "reconsider the ranking formula" in turn.thinking, \
        "the reasoning was thrown away"


@pytest.mark.real_summariser
def test_the_summariser_is_watched_for_circling_too(monkeypatch):
    """The third generation path, and the third to be missed.

    The round loop was covered, then _conclude, and summarise_span — which goes
    straight to `stream` and never passes through `stream_complete` — was not.
    Measured on a build task: two notes ran to 27,606 characters, essentially
    the 7,864-token cap, at 340s each. That was 683 of the 765 seconds the run
    spent compacting, against 40s for the two notes that behaved.

    frequency_penalty does not catch it: it stops verbatim repetition, and this
    is the paraphrasing kind.
    """
    from miniharness import context
    from miniharness.provider import AssistantTurn, TextChunk

    circle = ("2. Changed: nothing yet.\n"
              "Actually, let me restate what has changed.\n") * 400

    def fake_stream(model, system, messages, schemas, config):
        for i in range(0, len(circle), 400):
            yield TextChunk(circle[i:i + 400])
        raise AssertionError("the summariser was never interrupted")

    monkeypatch.setattr("miniharness.provider.stream", fake_stream)
    note = context.summarise_span([{"role": "user", "content": "go"}],
                                  "local", "sys", {"llama_ctx": 131072})
    assert note, "an interrupted summariser must still return what it wrote"
    assert "Changed" in note, "the partial note was discarded"
    assert len(note) < len(circle), "it ran to the cap despite the guard"
