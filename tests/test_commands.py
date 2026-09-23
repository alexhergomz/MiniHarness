"""The slash-command layer.

Every command in the REPL, exercised through the same entry point the REPL uses.
This layer had no coverage at all: it is the part of the harness a user touches
most directly, and the one where a mistake silently corrupts conversation state
rather than raising.
"""

from __future__ import annotations

import pytest

from miniharness import config as cfg_mod
from miniharness import context, loop, session
from miniharness.__main__ import handle_command


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated config/session home, plus a fresh state and config."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(cfg_mod, "HOME", home)
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", home / "config.toml")
    monkeypatch.setattr(session, "SESSIONS", home / "sessions")
    cfg = dict(cfg_mod.DEFAULTS)
    cfg["_cwd"] = str(tmp_path)
    state = loop.State(system="sys", messages=[])
    return state, cfg, tmp_path


def run(cmd, env):
    state, cfg, _ = env
    return handle_command(cmd, state, cfg, context.FileTracker())


# ── Control flow ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cmd", ["/quit", "/exit"])
def test_quit_ends_the_repl(cmd, env):
    assert run(cmd, env) is False


@pytest.mark.parametrize("cmd", ["/help", "/model", "/config", "/compact", "/undo"])
def test_every_command_keeps_the_repl_alive(cmd, env):
    assert run(cmd, env) is True


def test_unknown_command_does_not_crash(env):
    assert run("/nonsense", env) is True


def test_a_command_with_trailing_whitespace_still_parses(env):
    state, cfg, _ = env
    run("/model   gpt-4o   ", env)
    assert cfg["model"] == "gpt-4o"


# ── /model ──────────────────────────────────────────────────────────────────
def test_model_sets_and_persists(env):
    state, cfg, _ = env
    run("/model ollama/qwen3-coder", env)
    assert cfg["model"] == "ollama/qwen3-coder"
    assert "ollama/qwen3-coder" in cfg_mod.CONFIG_PATH.read_text()


def test_bare_model_command_does_not_clobber_the_setting(env):
    state, cfg, _ = env
    cfg["model"] = "gpt-4o"
    run("/model", env)
    assert cfg["model"] == "gpt-4o"


# ── /config ─────────────────────────────────────────────────────────────────
def test_config_sets_with_type_coercion(env):
    state, cfg, _ = env
    run("/config max_turns=7", env)
    assert cfg["max_turns"] == 7 and isinstance(cfg["max_turns"], int)
    run("/config accept_all=true", env)
    assert cfg["accept_all"] is True


def test_config_rejects_an_unknown_key_without_crashing(env):
    state, cfg, _ = env
    assert run("/config not_a_real_key=1", env) is True
    assert "not_a_real_key" not in cfg


def test_config_only_persists_non_defaults(env):
    state, cfg, _ = env
    run("/config max_turns=7", env)
    text = cfg_mod.CONFIG_PATH.read_text()
    assert "max_turns = 7" in text
    # temperature was never changed, so it should not be written out
    assert "temperature" not in text


def test_turbo_kv_passes_through_config(env):
    """The TurboQuant switch: a string the harness never validates or rewrites."""
    state, cfg, _ = env
    run("/config llama_kv_quant=turbo3", env)
    assert cfg["llama_kv_quant"] == "turbo3"


# ── /compact ────────────────────────────────────────────────────────────────
def test_compact_shrinks_a_long_conversation(env):
    state, cfg, _ = env
    state.messages = [{"role": "user", "content": "task"}]
    for i in range(40):
        state.messages.append({"role": "assistant", "content": "",
                               "tool_calls": [{"id": str(i), "type": "function",
                                               "function": {"name": "Bash",
                                                            "arguments": "{}"}}]})
        state.messages.append({"role": "tool", "tool_call_id": str(i), "content": "x" * 4000})
    before = context.estimate_tokens(state.messages)
    run("/compact", env)
    assert context.estimate_tokens(state.messages) < before
    assert state.messages[0]["content"] == "task"


def test_compact_on_an_empty_conversation_is_harmless(env):
    state, _, _ = env
    assert run("/compact", env) is True
    assert state.messages == []


# ── /think ──────────────────────────────────────────────────────────────────
def test_thinking_is_hidden_by_default():
    assert cfg_mod.DEFAULTS["show_thinking"] is False


def test_think_toggles_live_display_and_persists(env):
    state, cfg, _ = env
    run("/think on", env)
    assert cfg["show_thinking"] is True
    assert "show_thinking = true" in cfg_mod.CONFIG_PATH.read_text()
    run("/think off", env)
    assert cfg["show_thinking"] is False


def test_think_expands_the_last_turns_reasoning(env, capsys):
    state, cfg, _ = env
    state.last_thinking = "first I considered the divide function"
    run("/think", env)
    assert "divide function" in capsys.readouterr().out


def test_think_with_nothing_captured_says_so(env, capsys):
    state, cfg, _ = env
    state.last_thinking = ""
    run("/think", env)
    assert "No reasoning" in capsys.readouterr().out


# ── /undo ───────────────────────────────────────────────────────────────────
def test_undo_reverts_the_last_write(env, tmp_path):
    from miniharness import tools
    state, cfg, work = env
    f = work / "a.py"
    f.write_text("before\n")
    tracker = context.FileTracker()
    tracker.mark_read(str(f))
    tools.dispatch("Write", {"file_path": "a.py", "content": "after"}, cfg, tracker)
    assert f.read_text() == "after"
    run("/undo", env)
    assert f.read_text() == "before\n"


def test_undo_with_nothing_to_undo(env):
    from miniharness import tools
    tools._UNDO.clear()
    assert run("/undo", env) is True


# ── /resume ─────────────────────────────────────────────────────────────────
def test_resume_restores_the_message_history(env):
    state, cfg, _ = env
    msgs = [{"role": "user", "content": "earlier task"},
            {"role": "assistant", "content": "done"}]
    session.append("proj-20260101-000000", {"messages": msgs})
    run("/resume proj-20260101-000000", env)
    assert state.messages == msgs


def test_resume_with_no_argument_picks_the_most_recent(env):
    state, cfg, _ = env
    session.append("old-20260101-000000", {"messages": [{"role": "user", "content": "old"}]})
    session.append("new-20260102-000000", {"messages": [{"role": "user", "content": "new"}]})
    run("/resume", env)
    assert state.messages[0]["content"] == "new"


def test_resume_of_a_missing_session_leaves_state_untouched(env):
    state, cfg, _ = env
    state.messages = [{"role": "user", "content": "keep me"}]
    run("/resume does-not-exist", env)
    assert state.messages == [{"role": "user", "content": "keep me"}]


def test_resume_survives_a_corrupt_transcript_line(env):
    state, cfg, _ = env
    session.SESSIONS.mkdir(parents=True, exist_ok=True)
    p = session.SESSIONS / "broken.jsonl"
    p.write_text('{"messages": [{"role":"user","content":"good"}]}\n'
                 '{ this is not json\n')
    run("/resume broken", env)
    assert state.messages[0]["content"] == "good"


# ── /research ───────────────────────────────────────────────────────────────
def test_research_with_no_argument_lists_instead_of_running(env, monkeypatch):
    from miniharness import research
    called = []
    monkeypatch.setattr(research, "run", lambda *a, **k: called.append(a))
    assert run("/research", env) is True
    assert called == [], "bare /research must list, not spawn a run"


def test_research_failure_is_reported_not_raised(env, monkeypatch):
    from miniharness import research

    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(research, "run", boom)
    assert run("/research why is the sky blue", env) is True


def test_changing_the_window_on_a_running_server_restarts_it(monkeypatch, tmp_path):
    """Every budget is a share of llama_ctx, so a config claiming 64k against a
    server serving 32k puts every request ~10k tokens over the real limit. The
    harness suggests exactly this change when a start fails for want of VRAM."""
    from miniharness import __main__ as cli, context, loop, server
    from miniharness.config import DEFAULTS

    calls = {"stop": 0, "ensure": 0}
    monkeypatch.setattr(server, "live_ctx", lambda cfg, timeout=2.0: 32768)
    monkeypatch.setattr(server, "stop", lambda: calls.__setitem__("stop", calls["stop"] + 1))
    monkeypatch.setattr(server, "ensure", lambda cfg: calls.__setitem__("ensure", calls["ensure"] + 1) or True)

    cfg = dict(DEFAULTS)
    cfg.update({"_cwd": str(tmp_path), "model": "local", "llama_ctx": 32768})
    st = loop.State(system="s", messages=[])
    cli.handle_command("/config llama_ctx=65536", st, cfg, context.FileTracker())
    assert cfg["llama_ctx"] == 65536
    assert calls["stop"] == 1 and calls["ensure"] == 1, (
        "server was left serving the old window")


def test_an_unchanged_window_does_not_restart_the_server(monkeypatch, tmp_path):
    from miniharness import __main__ as cli, context, loop, server
    from miniharness.config import DEFAULTS
    calls = {"stop": 0}
    monkeypatch.setattr(server, "live_ctx", lambda cfg, timeout=2.0: 32768)
    monkeypatch.setattr(server, "stop", lambda: calls.__setitem__("stop", calls["stop"] + 1))
    cfg = dict(DEFAULTS)
    cfg.update({"_cwd": str(tmp_path), "model": "local", "llama_ctx": 1})
    st = loop.State(system="s", messages=[])
    cli.handle_command("/config llama_ctx=32768", st, cfg, context.FileTracker())
    assert calls["stop"] == 0


def test_a_remote_model_is_never_restarted(monkeypatch, tmp_path):
    """Only a locally managed server can be restarted; touching a remote
    provider's config must not try."""
    from miniharness import __main__ as cli, context, loop, server
    from miniharness.config import DEFAULTS
    calls = {"stop": 0}
    monkeypatch.setattr(server, "stop", lambda: calls.__setitem__("stop", calls["stop"] + 1))
    cfg = dict(DEFAULTS)
    cfg.update({"_cwd": str(tmp_path), "model": "gpt-4o", "llama_ctx": 8192})
    st = loop.State(system="s", messages=[])
    cli.handle_command("/config llama_ctx=65536", st, cfg, context.FileTracker())
    assert calls["stop"] == 0


# ── a long turn must not look like a hang ───────────────────────────────────
def test_a_long_think_reports_progress_not_a_static_line(monkeypatch, capsys):
    """`thinking…` on its own is indistinguishable from a hang.

    It was diagnosed as one twice. Until now the continuation notices bounded
    the silence — they fired every time the reply cap was hit — but the cap is
    gone (§2.8), so nothing else marks time during a multi-minute deliberation.
    The chunks are already arriving; counting them needs no server-specific
    endpoint and works for any OpenAI-compatible provider.
    """
    from miniharness import __main__ as m
    from miniharness import loop
    from miniharness.provider import ThinkChunk

    ticks = iter([0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    monkeypatch.setattr(m._time, "monotonic", lambda: next(ticks, 60.0))
    monkeypatch.setattr(loop, "run",
                        lambda *a, **k: iter([ThinkChunk("x" * 4000),
                                              ThinkChunk("y" * 4000)]))

    state = loop.State(messages=[{"role": "user", "content": "go"}])
    m.run_turn(state, {"model": "local", "show_thinking": False}, None)

    out = capsys.readouterr().out
    assert "thinking…" in out
    assert "tokens" in out, "no progress counter during a long think"
    assert "2,000 tokens" in out, f"counter did not accumulate: {out!r}"


def test_the_progress_counter_stays_out_of_the_way_when_reasoning_is_shown(
        monkeypatch, capsys):
    """With /think on, the reasoning itself is the progress indicator."""
    from miniharness import __main__ as m
    from miniharness import loop
    from miniharness.provider import ThinkChunk

    monkeypatch.setattr(loop, "run",
                        lambda *a, **k: iter([ThinkChunk("deliberating")]))
    state = loop.State(messages=[{"role": "user", "content": "go"}])
    m.run_turn(state, {"model": "local", "show_thinking": True}, None)

    out = capsys.readouterr().out
    assert "deliberating" in out
    assert "tokens," not in out


# ── Transcript and REPL conveniences ────────────────────────────────────────
def test_a_diff_marks_only_the_words_that_changed():
    """An edited line shows what changed inside it, not just that it changed."""
    from miniharness import preview
    rows = preview.diff_rows("a = 1\nb = total / (n - 1)\nc = 3\n",
                             "a = 1\nb = total / n\nc = 3\n")
    dels = [r for r in rows if r.kind == "del"]
    adds = [r for r in rows if r.kind == "add"]
    assert [r.old_no for r in dels] == [2] and [r.new_no for r in adds] == [2]
    changed = "".join(dels[0].text[a:b] for a, b in dels[0].spans)
    assert "(" in changed and "- 1)" in changed and "total" not in changed


def test_an_applied_change_is_shown_even_when_nothing_asked(tmp_path, monkeypatch, capsys):
    """Under --accept-all there is no prompt, so the transcript is the only
    place a change can be seen. It used to be a one-line tool result."""
    from miniharness import __main__ as m
    from miniharness import context, loop
    from miniharness.provider import AssistantTurn

    f = tmp_path / "stats.py"
    f.write_text("def mean(xs):\n    return sum(xs) / (len(xs) - 1)\n")
    turns = [AssistantTurn(text="", finish_reason="tool_calls", tool_calls=[
                 {"id": "1", "name": "Edit", "input": {
                     "file_path": "stats.py",
                     "old_string": "(len(xs) - 1)", "new_string": "len(xs)"}}]),
             AssistantTurn(text="fixed", finish_reason="stop")]
    monkeypatch.setattr(loop, "stream_complete", lambda *a, **k: iter([turns.pop(0)]))
    tracker = context.FileTracker()
    tracker.mark_read(str(f))
    state = loop.State(messages=[{"role": "user", "content": "fix"}])
    m.run_turn(state, {"model": "local", "_cwd": str(tmp_path), "accept_all": True},
               tracker)

    out = capsys.readouterr().out
    assert "Updated stats.py with 1 addition and 1 removal" in out
    assert "- " in out and "return sum(xs) / len(xs)" in out
    assert "1 file changed" in out and "/rewind" in out, "the undo was not mentioned"


def test_a_command_result_shows_its_verdict_not_its_progress_bar():
    from miniharness import __main__ as m
    head, n = m._result_line("Bash", "....   [100%]\n\n4 passed in 0.12s")
    assert head == "4 passed in 0.12s" and n == 2
    head, _ = m._result_line("Bash", "F [100%]\n1 failed in 0.1s\n[exit 1]")
    assert head == "1 failed in 0.1s  [exit 1]"


def test_an_at_file_is_attached_through_the_jailed_read(tmp_path):
    """Attached with the same Read the model would use — so the jail applies,
    and the file counts as read and can be edited on the first call."""
    from miniharness import __main__ as m
    from miniharness import context

    (tmp_path / "a.py").write_text("x = 1\n")
    cfg = {"_cwd": str(tmp_path)}
    tracker = context.FileTracker()
    msg = m.compose_message("look at @a.py and @nope.py", cfg, tracker)
    assert "Contents of `a.py`" in msg and "x = 1" in msg
    assert "nope.py`" not in msg.split("look at")[1].split("\n", 1)[1]
    assert tracker.has_read(str(tmp_path / "a.py"))

    out = m.compose_message("@/etc/shadow", cfg, context.FileTracker())
    assert "Contents of" not in out, "an @mention went around the jail"


def test_shell_output_waits_for_the_next_message(tmp_path):
    """Sent alone it would be a user message with no question in it, and two
    user messages in a row break chat templates that require alternation."""
    from miniharness import __main__ as m
    cfg = {"_cwd": str(tmp_path)}
    m._PENDING_SHELL.clear()
    m.run_shell("echo from-the-user", cfg)
    msg = m.compose_message("what does that mean?", cfg, None)
    assert msg.index("from-the-user") < msg.index("what does that mean?")
    assert m.compose_message("next", cfg, None) == "next", "sent twice"


def test_completion_offers_commands_and_paths(tmp_path):
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.document import Document
    from miniharness import __main__ as m

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "stats.py").write_text("")
    (tmp_path / "__pycache__").mkdir()
    c = m.make_completer({"_cwd": str(tmp_path)})
    texts = lambda s: [x.text for x in c.get_completions(Document(s), None)]  # noqa: E731
    assert texts("/di") == ["/diff"]
    assert texts("see @s") == ["@src/"]
    assert texts("@src/st") == ["@src/stats.py"]
    assert texts("@_") == [], "noise directories should not be offered"


def test_clear_starts_over_but_keeps_the_old_conversation(tmp_path):
    from miniharness import __main__ as m
    from miniharness import context, loop
    state = loop.State(system="sys", messages=[{"role": "user", "content": "x"}],
                       session_id="old-one")
    tracker = context.FileTracker()
    tracker.mark_read("/somewhere/a.py")
    m.handle_command("/clear", state, {"_cwd": str(tmp_path)}, tracker)
    assert state.messages == [] and state.system == "sys"
    assert state.session_id != "old-one"
    assert not tracker.has_read("/somewhere/a.py")


def test_sampling_is_the_model_cards_unless_the_user_overrules_it(monkeypatch):
    """The harness sent temperature 0.3 on every request, against Qwen's
    documented 0.6 for coding — and low temperature is what drove the
    repetition loops. The model's authors decide; the user may overrule."""
    from miniharness import config as cfg_mod
    from miniharness import server

    cfg = dict(cfg_mod.DEFAULTS, llama_model_path="/m/Qwen3.5-4B-Q4_K_M.gguf",
               llama_host="127.0.0.1", llama_port=8890, prefix_checkpoint=False)
    args = server.build_args(cfg)
    assert args[args.index("--temp") + 1] == "0.6"
    assert args[args.index("--top-k") + 1] == "20"

    unknown = server.build_args(dict(cfg, llama_model_path="/m/mystery-7b.gguf"))
    assert "--temp" not in unknown, "a guess passed off as the model's own"

    captured = {}

    def fake_connect(url, headers, payload, config):
        captured.update(payload)
        raise RuntimeError("stop")

    from miniharness import provider
    monkeypatch.setattr(provider, "_connect", fake_connect)
    for c in (dict(cfg_mod.DEFAULTS, model="local", max_retries=0),
              dict(cfg_mod.DEFAULTS, model="local", max_retries=0, temperature=0.2)):
        captured.clear()
        try:
            list(provider.stream("local", "sys", [], [], c))
        except Exception:
            pass
        assert captured.get("messages"), "the request was never built"
        if c["temperature"] == "":
            assert "temperature" not in captured, "the harness overrode the model"
        else:
            assert captured["temperature"] == 0.2


def test_the_approval_menu_without_a_terminal_takes_a_number(monkeypatch):
    """Piped input gets a numbered prompt; the old `[y]es / [n]o` text was
    swallowed by Rich markup and rendered as "es / o"."""
    from miniharness import __main__ as m
    for typed, want in (("", 0), ("2", 1), ("3", 2), ("y", 0), ("a", 1), ("nope", 2)):
        monkeypatch.setattr(m.console, "input", lambda *a, _t=typed, **k: _t)
        assert m._choose_by_number("Allow?", ["Yes", "Always", "No"], cancel=2) == want
