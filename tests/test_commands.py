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
