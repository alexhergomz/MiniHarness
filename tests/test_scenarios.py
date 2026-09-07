"""Whole trajectories, judged the way an engineer would judge them.

The unit tests check mechanisms. These check the thing that actually matters:
after the harness has done its budgeting, can the model still finish the job?

Each scenario is a plausible agent transcript pushed past the budget. The
assertions are about capability, not bytes — does it still know the task, the
last failure, what it already changed, and how to get back anything that was
taken away.
"""
from __future__ import annotations

import json

import pytest

from miniharness import context


# ── building trajectories ──────────────────────────────────────────────────
def call(cid: str, name: str, args: dict, result: str) -> list[dict]:
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args, sort_keys=True)}}]},
        {"role": "tool", "tool_call_id": cid, "content": result},
    ]


def says(text: str) -> dict:
    return {"role": "user", "content": text}


def find(out: list[dict], cid: str) -> str:
    for m in out:
        if m.get("tool_call_id") == cid:
            return str(m.get("content") or "")
    return ""


def dropped(out: list[dict], cid: str) -> bool:
    return not any(m.get("tool_call_id") == cid for m in out)


def text_of(out: list[dict]) -> str:
    return "\n".join(str(m.get("content") or "") for m in out)


SRC = "def parse(s):\n" + "\n".join(f"    step_{i}(s)" for i in range(120))
TRACE = ("Traceback (most recent call last):\n"
         '  File "tests/test_parse.py", line 12, in test_round_trip\n'
         "    assert parse(x) == x\nAssertionError: 'a' != 'b'\n"
         "FAILED tests/test_parse.py::test_round_trip\n") + "detail\n" * 400


# ── scenario 1: the debug loop ─────────────────────────────────────────────
def _debug_loop() -> list[dict]:
    """Read, test, fix, test again — the commonest long trajectory there is."""
    msgs = [says("The round-trip test in tests/test_parse.py fails. Fix it.")]
    msgs += call("t1", "Bash", {"command": "pytest -x"}, TRACE)
    msgs += call("r1", "Read", {"file_path": "src/parse.py"}, SRC * 3)
    msgs += call("e1", "Edit", {"file_path": "src/parse.py",
                                "old_string": "step_7(s)", "new_string": "step_7(s.strip())"},
                 "Edited src/parse.py (1 replacement)")
    msgs += call("t2", "Bash", {"command": "pytest -x"},
                 "FAILED tests/test_parse.py::test_round_trip - IndexError: list index "
                 "out of range\n" + "detail\n" * 400)
    msgs += call("r2", "Read", {"file_path": "src/tokens.py"}, SRC * 3)
    return msgs


def test_the_task_statement_always_survives():
    out, _ = context.compact_with_model(_debug_loop(), 400, "local", "sys", {})
    assert "round-trip test" in text_of(out)


def test_the_edit_that_was_already_made_is_never_forgotten():
    """Losing this makes the model redo work it has already done, or undo it."""
    out, _ = context.compact_with_model(_debug_loop(), 500, "local", "sys", {})
    assert "step_7" in text_of(out) or "Edited src/parse.py" in text_of(out)


def test_the_history_stays_well_formed_at_every_budget():
    """A malformed history is a 400, which is worse than any amount of loss."""
    msgs = _debug_loop()
    for budget in (200, 400, 800, 1500, 3000, 10_000):
        out, _ = context.compact_with_model([dict(m) for m in msgs], budget, "local", "sys", {})
        pending: list[str] = []
        prev = None
        for m in out:
            if m["role"] == "assistant":
                assert not pending, f"unanswered tool_calls at budget {budget}"
                pending = [t["id"] for t in m.get("tool_calls", [])]
            elif m["role"] == "tool":
                assert m["tool_call_id"] in pending, f"orphan tool msg at {budget}"
                pending.remove(m["tool_call_id"])
            elif m["role"] == "user":
                assert prev != "user", f"user->user at budget {budget}"
            prev = m["role"]
        assert not pending, f"ends with unanswered tool_calls at {budget}"


# ── scenario 2: a constraint given mid-conversation ────────────────────────
def test_an_instruction_given_late_is_not_lost():
    """Governance decay: the constraint arrives at turn 5, the run ends without
    it, and nothing in the transcript shows it was ever dropped."""
    msgs = [says("Refactor the parser.")]
    for i in range(6):
        msgs += call(f"r{i}", "Read", {"file_path": f"src/m{i}.py"}, SRC * 2)
    msgs += [says("When you finish, write the token QUUX to DONE.txt.")]
    for i in range(6, 12):
        msgs += call(f"r{i}", "Read", {"file_path": f"src/m{i}.py"}, SRC * 2)
    out, _ = context.compact_with_model(msgs, 900, "local", "sys", {})
    assert "QUUX" in text_of(out), "the mid-conversation constraint was evicted"


def test_older_instructions_are_given_up_only_once_a_newer_one_is_safe():
    """Older instructions are compacted with their context — kept verbatim they
    would refer to files and errors no longer in the transcript. But nothing is
    given up while the most recent instruction is itself at risk."""
    msgs = [says("Original task with the required output format.")]
    for i in range(120):
        msgs += [says(f"rule {i}: " + "r" * 400)]
        msgs += call(f"c{i}", "Read", {"file_path": f"f{i}.py"}, SRC)
    msgs += [says("Final instruction: write the token QUUX to DONE.txt.")]
    out, _ = context.compact_with_model(msgs, 900, "local", "sys", {})
    blob = text_of(out)
    assert "QUUX" in blob, "the most recent instruction was lost"


# ── scenario 3: stale reads ────────────────────────────────────────────────
def test_a_large_useless_result_is_evicted_before_a_useful_one():
    msgs = [says("Find the handler")]
    msgs += call("g1", "Grep", {"pattern": "zzz"}, "No matches found\n" + "\n" * 3000)
    msgs += call("t1", "Bash", {"command": "pytest"}, "FAILED test_x - assert 1 == 2\n"
                 + "detail\n" * 300)
    out, _ = context.compact_with_model(msgs, 700, "local", "sys", {})
    assert "FAILED test_x" in find(out, "t1"), "kept the empty search over the failure"


# ── scenario 4: nothing is recoverable ─────────────────────────────────────
def test_among_commands_the_failures_outlive_the_successes():
    msgs = [says("Get the build green.")]
    for i in range(5):
        msgs += call(f"ok{i}", "Bash", {"command": f"echo step{i}"},
                     "step done\n" + "output\n" * 300)
    msgs += call("bad", "Bash", {"command": "make"},
                 "error: undefined reference to `foo'\n" + "line\n" * 300)
    out, _ = context.compact_with_model(msgs, 900, "local", "sys", {})
    assert "undefined reference" in find(out, "bad"), "gave up the failure first"


# ── scenario 5: pressure that never ends ───────────────────────────────────
@pytest.mark.parametrize("budget", [300, 700, 1500, 4000])
def test_compaction_always_reaches_budget_on_a_real_trajectory(budget):
    """It must fit — down to the one thing that cannot be made smaller.

    Without elision a single tool result larger than the entire budget stays
    whole: truncating it would be elision under another name. In a real session
    `tool_output_share` keeps results far below the history budget, so this
    floor is only reachable at budgets no session actually uses.
    """
    out, _ = context.compact_with_model(_debug_loop(), budget, "local", "sys", {})
    total = context.estimate_tokens(out)
    biggest = max(context.estimate_tokens([m]) for m in out)
    assert total <= budget or total - biggest <= budget, (
        f"{total} > {budget}, and still over once the single largest message "
        f"({biggest} tokens) is set aside")


def test_repeated_compaction_converges():
    """Compaction runs every few turns for the life of a task. If each pass
    shaved a little more, a long run would grind the history to nothing."""
    msgs = _debug_loop()
    out, _ = context.compact_with_model(msgs, 1200, "local", "sys", {})
    size = context.estimate_tokens(out)
    for _ in range(6):
        out, _ = context.compact_with_model(out, 1200, "local", "sys", {})
    assert context.estimate_tokens(out) == size, "compaction is not idempotent"


# ── scenario 6: the model must be able to see what it has already done ─────
def test_the_working_set_reports_repeats_so_a_loop_is_visible():
    """A repeat count is the cheapest signal that the model is going in circles,
    and the one thing it cannot observe about itself."""
    msgs = [says("build it")]
    for i in range(4):
        msgs += call(f"r{i}", "Read", {"file_path": "CMakeLists.txt"}, SRC)
    ws = context.working_set(msgs)
    assert "[4x]" in ws
    assert "act on what you have" in ws


def test_the_working_set_names_the_outcome_of_each_command():
    msgs = [says("get it green")]
    msgs += call("b1", "Bash", {"command": "make"}, "gcc ...\nmake: *** [all] Error 2")
    msgs += call("b2", "Bash", {"command": "ls"}, "a.c\nb.c")
    ws = context.working_set(msgs)
    assert "Error 2" in ws, "the failure's outcome must be visible"
    assert "make" in ws


def test_the_working_set_is_bounded():
    msgs = [says("go")]
    for i in range(400):
        msgs += call(f"r{i}", "Read", {"file_path": f"src/really/long/path/module_{i}.py"}, SRC)
        msgs += call(f"b{i}", "Bash", {"command": f"echo {i} " + "x" * 200}, "ok")
    ws = context.working_set(msgs)
    assert context.estimate_tokens([{"role": "user", "content": ws}]) < 1600


def test_the_working_set_is_empty_before_anything_happens():
    assert context.working_set([says("do a thing")]) is None


def test_the_record_of_what_changed_survives_compaction():
    """Compaction must not be able to reach the record of work already done.

    It could: the edits lived in the span that got replaced, so the working set
    went from naming two changed files to returning None, while the note
    covering that span said "nothing changed yet". Two sources agreeing the
    model had done nothing is how one scenario spent 31 Reads against 3 Edits.
    """
    msgs = [says("wire the setting through")]
    ledger = []
    for cid, name, args, body in (
            ("c1", "Edit", {"file_path": "config.py", "old_string": "compact_to"}, "edited"),
            ("c2", "Edit", {"file_path": "tools.py", "old_string": "lines ="}, "edited"),
            ("b1", "Bash", {"command": "pytest -q"}, "1 failed, 344 passed"),
    ):
        msgs += call(cid, name, args, body)
        ledger.append((cid, name, args, body))
    msgs += call("r1", "Read", {"file_path": "big.py"}, "x" * 9000)
    ledger.append(("r1", "Read", {"file_path": "big.py"}, "x" * 200))

    before = context.working_set(msgs)
    assert "config.py" in before and "tools.py" in before

    out, _ = context.compact_with_model(list(msgs), 200, "local", "sys", {})
    lost = context.working_set(out) or ""
    assert "config.py" not in lost and "tools.py" not in lost, (
        "the edits really were in the replaced span")

    after = context.working_set(out, ledger=ledger)
    assert "config.py" in after and "tools.py" in after
    # The outcome of a command that ran is a fact worth keeping too.
    assert "1 failed, 344 passed" in after


def test_the_ledger_does_not_double_count_a_call_still_in_the_transcript():
    """A call in both places is one call, or every repeat count is inflated."""
    msgs = [says("look")] + call("r1", "Read", {"file_path": "a.py"}, "body")
    ledger = [("r1", "Read", {"file_path": "a.py"}, "body")]
    ws = context.working_set(msgs, ledger=ledger)
    assert "Read a.py" in ws
    assert "2x" not in ws


# ── scenario 7: model-written notes instead of elision ─────────────────────
def test_model_compaction_preserves_governance(monkeypatch):
    monkeypatch.setattr(context, "summarise_span",
                        lambda *a, **k: "Learned X. Edited src/parse.py. Still failing.")
    msgs = _debug_loop()
    msgs.insert(5, says("Also: write QUUX to DONE.txt when finished."))
    out, _ = context.compact_with_model(msgs, 600, "local", "sys", {})
    blob = text_of(out)
    assert "QUUX" in blob, "a user instruction was summarised away"
    assert "round-trip test" in blob


def test_model_compaction_produces_a_valid_history(monkeypatch):
    monkeypatch.setattr(context, "summarise_span", lambda *a, **k: "Notes.")
    msgs = _debug_loop()
    for budget in (300, 600, 1200):
        out, _ = context.compact_with_model([dict(m) for m in msgs], budget,
                                            "local", "sys", {})
        pending = []
        prev = None
        for m in out:
            if m["role"] == "assistant":
                assert not pending, f"unanswered tool_calls at {budget}"
                pending = [t["id"] for t in m.get("tool_calls", [])]
            elif m["role"] == "tool":
                assert m["tool_call_id"] in pending, f"orphan tool msg at {budget}"
                pending.remove(m["tool_call_id"])
            elif m["role"] == "user":
                assert prev != "user", f"user->user at {budget}"
            prev = m["role"]
        assert not pending


@pytest.mark.real_summariser
def test_the_summariser_is_greedy_and_the_agent_is_not(monkeypatch):
    """Deterministic decode belongs to the mechanical call only. Applied to the
    agent it changes the thing under test — and greedy decoding makes a small
    model repeat itself, which looks exactly like the blindness the summary is
    meant to cure."""
    seen = {}

    def fake_stream(model, system, messages, schemas, config):
        seen["cfg"] = config
        class T:
            text = "notes"
        yield T()

    import miniharness.provider as P
    monkeypatch.setattr(P, "stream", fake_stream)
    context.summarise_span([says("go")], "local", "sys", {"temperature": 0.3})
    assert seen["cfg"]["temperature"] == 0.0, "summariser must be greedy"
    assert seen["cfg"]["disable_thinking"] is True

    # and the caller's own config is untouched
    cfg = {"temperature": 0.3}
    context.summarise_span([says("go")], "local", "sys", cfg)
    assert cfg["temperature"] == 0.3, "summariser leaked its settings to the agent"


@pytest.mark.real_summariser
def test_the_summariser_does_not_wear_the_agents_system_prompt(monkeypatch):
    """Handed the agent's prompt the model stays in coding-agent voice and
    writes activity instead of values — measured: "Installed cmake via apt-get"
    with the version dropped, which the next reader confabulated as 3.28.2."""
    seen = {}

    def fake_stream(model, system, messages, schemas, config):
        seen["system"] = system
        class T:
            text = "notes"
        yield T()

    import miniharness.provider as P
    monkeypatch.setattr(P, "stream", fake_stream)
    context.summarise_span([says("go")], "local",
                           "You are a coding agent working in a terminal.", {})
    assert "coding agent working in a terminal" not in seen["system"]
    assert "extract facts" in seen["system"].lower()
    assert "never categories" in seen["system"].lower() or \
           "concrete values" in seen["system"].lower()


def test_model_compaction_reaches_its_budget(monkeypatch):
    """It must fit, for every budget a real session can ask for.

    The floor is one message: without elision, a single tool result larger than
    the entire budget cannot be made smaller, and truncating it would be
    elision under another name. ``tool_output_share`` is what keeps results far
    below the history budget in practice, so the guarantee is stated against
    that floor rather than an arbitrary number.
    """
    monkeypatch.setattr(context, "summarise_span", lambda *a, **k: "note " * 40)
    for budget in (400, 800, 1600):
        out, _ = context.compact_with_model(_debug_loop(), budget, "local", "sys", {})
        total = context.estimate_tokens(out)
        biggest = max(context.estimate_tokens([m]) for m in out)
        assert total <= budget or total - biggest <= budget, (
            f"{total} > {budget}, and still over once the single largest "
            f"message ({biggest} tokens) is set aside")


@pytest.mark.real_summariser
def test_the_summariser_is_given_the_models_own_reasoning(monkeypatch):
    """Reasoning is stripped from history by design, so the transcript holds
    tool calls and results and nothing the model concluded. Measured on one
    task: 4,841 characters of reasoning produced, 96 characters of visible prose
    kept — and the resulting note was a list of function names and line numbers,
    because that was all there was to summarise. The model had diagnosed the bug
    and planned the fix one turn earlier, then re-derived it from scratch."""
    seen = {}

    def fake_stream(model, system, messages, schemas, config):
        seen["messages"] = messages
        class T:
            text = "notes"
        yield T()

    import miniharness.provider as P
    monkeypatch.setattr(P, "stream", fake_stream)
    context.summarise_span(
        [says("go")], "local", "sys", {},
        thinking=["The bug is that an empty old_string matches everywhere; "
                  "guard at the top of _edit."])
    blob = "\n".join(str(m.get("content") or "") for m in seen["messages"])
    assert "empty old_string matches everywhere" in blob, \
        "the summariser never saw what the model worked out"


def test_the_summary_prompt_asks_for_conclusions_not_just_observations():
    assert "worked out" in context.SUMMARISE or "diagnosis" in context.SUMMARISE


def test_reasoning_is_recorded_but_never_sent_back_as_context(tmp_path, monkeypatch):
    """It goes to the summariser, not into the model's own context — putting it
    back would cost tokens every turn and feed the model its own half-formed
    thinking."""
    from miniharness import loop
    from miniharness.provider import AssistantTurn

    turn = AssistantTurn(text="ok", finish_reason="stop")
    turn.thinking = "I concluded the parser drops whitespace at step 7."
    monkeypatch.setattr(loop, "stream_complete", lambda *a, **k: iter([turn]))
    st = loop.State(system="s", messages=[])
    st.add_user("go")
    list(loop.run(st, {"model": "local", "_cwd": str(tmp_path), "max_turns": 2,
                       "accept_all": True, "repo_map": False}, None, None))
    assert st.recent_thinking and "step 7" in st.recent_thinking[0]
    stored = "\n".join(str(m.get("content") or "") for m in st.messages)
    assert "step 7" not in stored, "reasoning leaked into the transcript"


def test_the_reasoning_record_is_bounded(tmp_path, monkeypatch):
    from miniharness import loop
    from miniharness.provider import AssistantTurn

    def make(i):
        t = AssistantTurn(text=f"turn {i}", finish_reason="stop")
        t.thinking = f"thought {i}"
        return t

    turns = [make(i) for i in range(12)]
    monkeypatch.setattr(loop, "stream_complete",
                        lambda *a, **k: iter([turns.pop(0)]))
    st = loop.State(system="s", messages=[])
    for i in range(12):
        st.add_user("go")
        list(loop.run(st, {"model": "local", "_cwd": str(tmp_path),
                           "max_turns": 1, "accept_all": True,
                           "repo_map": False}, None, None))
    assert len(st.recent_thinking) <= 6, len(st.recent_thinking)


def test_compaction_does_not_strip_mid_turn_reasoning():
    """Compaction must not take the model's plan out from under an open turn.

    An assistant turn stays open until its tool calls are answered, and the
    server renders the reasoning of *every* step of an open turn (measured:
    a plan stated in step 1 was still acted on after two intervening tool
    calls). Compaction fires mid-turn, so if it dropped reasoning_content while
    shrinking tool output, the model would lose the plan exactly when it needed
    it — the failure this whole mechanism exists to prevent.
    """
    msgs = [{"role": "user", "content": "do the thing"}]
    for i in range(14):
        msgs.append({"role": "assistant", "content": "",
                     "reasoning_content": f"PLAN-{i}: " + "detail " * 60,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "Read",
                                                  "arguments": '{"file_path":"f.py"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": "line of file content\n" * 200})

    before = context.estimate_tokens(msgs, "sys")
    out, freed = context.compact_with_model(list(msgs), int(before * 0.4), "local", "sys", {})

    assert freed > 0 and context.estimate_tokens(out, "sys") < before
    kept = [m for m in out if m.get("role") == "assistant" and m.get("tool_calls")]
    assert kept, "compaction removed the whole open turn"
    assert all(m.get("reasoning_content") for m in kept), \
        "compaction stripped reasoning from an open turn"
    for m in kept:
        for tc in m["tool_calls"]:
            assert any(t.get("tool_call_id") == tc["id"] for t in out), \
                "compaction orphaned a tool call"


def test_open_turn_reasoning_is_counted_against_the_budget():
    """Reasoning that reaches the model must be on the ruler.

    §1's whole family of overflow bugs came from budgeting the convenient part
    of the request instead of the sent part. Preserving reasoning re-opens that
    hole unless the estimator follows: an open turn's reasoning is rendered
    (measured 225 prompt tokens vs 173 for three steps), a finished turn's is
    not (91 vs 91).
    """
    big = "reasoning detail " * 400

    def convo(closed):
        msgs = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "reasoning_content": big,
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "Read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "data"}]
        if closed:
            msgs += [{"role": "assistant", "content": "done"},
                     {"role": "user", "content": "next question"}]
        return msgs

    open_turn = context.raw_chars(convo(False))
    bare = context.raw_chars([m for m in convo(False)
                              if not m.pop("reasoning_content", None) or True])
    assert open_turn > bare + len(big) - 1, "open-turn reasoning went uncounted"

    # Once the model has answered and the user has spoken again, the template
    # drops it, so charging for it would waste real context.
    closed = convo(True)
    assert context.raw_chars(closed) < len(big), \
        "a finished turn's reasoning is still being charged for"

