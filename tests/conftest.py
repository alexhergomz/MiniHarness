"""Shared fixtures.

Compaction is written by the model, so every test that compacts would
otherwise reach for a server. Stubbed by default with a note that carries the
kind of content a real one does — the specific facts, not an index — so tests
assert on the mechanism rather than on the model's prose.
"""
import pytest

from miniharness import checkpoint, context


@pytest.fixture(autouse=True)
def _stub_summariser(monkeypatch, request):
    if request.node.get_closest_marker("real_summariser"):
        return
    monkeypatch.setattr(
        context, "summarise_span",
        lambda span, *a, **k: (
            "1. Established: read src/parse.py and src/tokens.py; the round-trip "
            "test fails with IndexError. Edited src/parse.py: step_7(s) -> "
            "step_7(s.strip()).\n"
            "2. Changed: src/parse.py.\n"
            "3. Still wrong: IndexError: list index out of range.\n"
            "4. Next: read src/tokens.py around the index arithmetic."))


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Every test gets its own ~/.miniharness. Nothing may reach the real one.

    One test built a config with model="gpt-4o" and ran /config, which saves —
    and it wrote to the user's real config.toml. Every run of the suite reset
    their setup to a cloud model they do not use, and it was found only when
    the harness refused to start. Patching that one test would leave the next
    one free to do the same, so every path derived from HOME is redirected
    here, for every test, whether or not it looks like it writes anything.
    """
    from miniharness import config as cfg_mod
    from miniharness import research, server, session

    home = tmp_path_factory.mktemp("miniharness-home")
    monkeypatch.setattr(cfg_mod, "HOME", home)
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", home / "config.toml")
    monkeypatch.setattr(session, "SESSIONS", home / "sessions")
    monkeypatch.setattr(checkpoint, "STORE", home / "checkpoints")
    monkeypatch.setattr(server, "SLOT_DIR", home / "slots")
    monkeypatch.setattr(server, "LOG_PATH", home / "llama-server.log")
    monkeypatch.setattr(research, "WORKSPACES", home / "research")
    monkeypatch.setenv("MINIHARNESS_HOME", str(home))


@pytest.fixture(autouse=True)
def _no_server_tokenizer(monkeypatch):
    """Tests count with the estimate unless they set a tokenizer themselves —
    never with a real server that happens to be running on the machine."""
    from miniharness import server
    monkeypatch.setattr(server, "exact_counter", lambda config: None)
    context.use_tokenizer(None)
    yield
    context.use_tokenizer(None)


@pytest.fixture(autouse=True)
def _no_checkpoints(monkeypatch, request):
    """Checkpoints run `git add -A` over the agent's working directory.

    Left on, every loop test that writes a file would shell out to git and
    leave a store under the real ~/.miniharness. Off by default; the tests that
    are about checkpointing ask for it and point the store at a tmp_path.
    """
    if request.node.get_closest_marker("checkpoints"):
        return
    monkeypatch.setattr(checkpoint, "enabled", lambda cfg: False)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_summariser: the test is about summarise_span itself, so it must "
        "not be stubbed")
    config.addinivalue_line(
        "markers",
        "checkpoints: the test is about checkpointing, so leave it enabled")
