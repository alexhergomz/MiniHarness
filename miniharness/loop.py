"""The agent loop: stream, dispatch, repeat.

Small enough to hold in your head, which is the point. The only genuinely subtle
thing in here is truncation handling, and it is subtle for a reason — see
``_handle_truncation``.

The loop yields events rather than printing. That keeps it UI-free and testable:
``__main__.py`` renders the events, tests assert on them.
"""

from __future__ import annotations

import json
import time as _time
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

from . import checkpoint, tools
from .provider import (AssistantTurn, Continuing, StoppedCircling, TextChunk, ThinkChunk,
                       ToolDraft, stream_complete)

# The stub that keeps user/assistant alternation intact after a truncation.
TRUNCATION_STUB = "[output cut off at max_tokens]"
EMPTY_STUB = "[no output produced]"


# ── Events ──────────────────────────────────────────────────────────────────
@dataclass
class ToolStart:
    name: str
    params: dict
    round: int = 0          # which tool round of this turn, 1-based
    max_rounds: int = 0


@dataclass
class ToolEnd:
    name: str
    result: str
    denied: bool = False


@dataclass
class TurnDone:
    turn: AssistantTurn


@dataclass
class Notice:
    text: str


@dataclass
class Compacting:
    """A compaction is starting. It is a model call — tens of seconds of
    silence otherwise, which reads as a hang. The next event means it ended."""
    tokens: int


@dataclass
class State:
    """Everything that persists across one conversation."""
    system: str = ""
    messages: list[dict] = field(default_factory=list)
    continuations: int = 0
    empty_retries: int = 0
    # Last turn's reasoning. Held for "/think" only — never sent to the model.
    last_thinking: str = ""
    # A bounded record of what the model has recently worked out, for the
    # summariser.
    #
    # Reasoning itself now reaches the model the way it is meant to: attached to
    # the assistant message that owns the tool calls, for as long as that turn
    # is open (see AssistantTurn.to_message). This list is a separate thing —
    # it is handed to the *summariser*, which sees only the span being replaced.
    # Without it, notes came back as an index rather than a conclusion: measured
    # on one task, 4,841 characters of reasoning produced against 96 characters
    # of visible prose, so a summary of the visible transcript had nothing but
    # function names and line numbers to work from.
    recent_thinking: list = field(default_factory=list)
    # Every tool call this conversation has made: (id, name, args, short result).
    #
    # The working set used to be derived from `messages` alone, which meant
    # compaction erased the record of what had already been done along with the
    # bulk it was trying to reclaim — two files edited, one compaction, and the
    # model was told nothing had changed. This is the one record compaction must
    # not be able to reach, so it lives here instead of in the transcript.
    #
    # Append-only, and bounded by storing the reduced form: working_set only ever
    # needs the last few entries and a repeat count, never a result body.
    ledger: list = field(default_factory=list)
    compact_floor: int = 0        # size a failed compaction settled at
    # Which transcript this conversation appends to. Lives here rather than in
    # the REPL because /resume has to rebind it: otherwise turns after a resume
    # append to the new session and the resumed one silently stops growing.
    session_id: str = ""

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


def _strip_malformed(turn: AssistantTurn, msg: dict) -> list[dict]:
    """Drop tool calls whose arguments failed to parse, from turn *and* history.

    Leaving one in history guarantees a 400 on the next request: there would be
    a tool_call with no matching tool response.
    """
    valid = [tc for tc in turn.tool_calls if "_raw" not in tc.get("input", {})]
    if len(valid) != len(turn.tool_calls):
        turn.tool_calls = valid
        if valid:
            msg["tool_calls"] = [m for m in msg.get("tool_calls", [])
                                 if any(m["function"]["name"] == v["name"] for v in valid)]
        else:
            msg.pop("tool_calls", None)
    return valid


def _handle_truncation(turn: AssistantTurn, msg: dict, state: State, config: dict) -> str | None:
    """Apply the alternation fix. Returns a continuation hint, or None to stop.

    When ``max_tokens`` cuts a response mid-tool-call, stripping the malformed
    call can leave an assistant message with no text and no tool calls. The
    naive fix is to pop it — but popping produces two consecutive user messages
    once the continuation hint is appended, and that alternation violation makes
    models misbehave in ways that look like harness bugs: the local Qwen 9B
    spams unrelated tool calls, other models silently re-emit their previous
    output ("the harness repeated itself").

    So: never pop. Replace empty content with a stub, keeping history
    well-alternated as user -> assistant(stub) -> user(hint) -> assistant(retry).
    """
    if not turn.truncated:
        return None
    if not (turn.text or "").strip() and not turn.tool_calls:
        msg["content"] = TRUNCATION_STUB
        turn.text = TRUNCATION_STUB

    # Continuation itself is handled transparently in provider.stream_complete,
    # which merges the pieces into one assistant message. Reaching here means
    # that layer gave up — most usefully because it ran out of *window*, not
    # rounds. That is worth one more try from here rather than in the provider,
    # because `continue` re-enters the loop at `_compact_if_needed`: compaction
    # frees room first, and only then is the reply resumed.
    #
    # This gets its own budget rather than reusing `max_continuations`, which
    # stream_complete has already spent — sharing the key would silently double
    # the cap and hide which layer was doing the work.
    #
    # Deliberately no "be more concise": the cap is the harness's limitation,
    # and telling the model to write less to fit it is working against it.
    cap = int(config.get("max_truncation_resumes", 2) or 0)
    if cap <= 0 or state.continuations >= cap:
        return None
    state.continuations += 1
    return ("Your previous message is still incomplete. Continue from exactly "
            "where it stopped.")


def _handle_empty_turn(turn: AssistantTurn, msg: dict, state: State,
                       config: dict) -> str | None:
    """Nudge a model that produced nothing at all. Returns a hint, or None.

    Observed with Qwen3.5-4B on a multi-step task: it ran the tests, spent ~173
    tokens reasoning inside a <think> block, then emitted no visible text and no
    tool call. The loop treated that as "the model is finished" and stopped with
    the task half-done — the classic early-stopping failure, and indistinguishable
    from success from the outside.

    An empty turn is never a legitimate completion: a model that is done says so.
    So stub the message to keep alternation clean, then ask it to act. Bounded,
    because a model that cannot produce output will not start after ten tries.
    """
    if (turn.text or "").strip() or turn.tool_calls:
        return None

    msg["content"] = EMPTY_STUB
    turn.text = EMPTY_STUB

    cap = int(config.get("max_empty_retries", 2) or 0)
    if cap <= 0 or state.empty_retries >= cap:
        return None
    state.empty_retries += 1

    if turn.thinking:
        return ("You produced reasoning but no visible answer and no tool call. "
                "Stop reasoning and act now: either call a tool to make progress, "
                "or state your final answer directly.")
    return ("Your last response was empty. Continue the task: call a tool to "
            "make progress, or state your final answer directly.")


def _attach(messages: list[dict], text: str) -> list[dict]:
    """Add harness text to the end of a request without starting a new query.

    Mid-task the request ends with a tool result. The harness's notes — the
    repository map, the working set, the rounds left — used to follow it as a
    user message of their own. But chat templates decide which reasoning to
    keep by the *last user message*: Qwen3.5's renders an assistant step's
    reasoning only if it comes after the last real user query, and a note sent
    as a user message is one. So every request dropped the model's reasoning
    for every earlier step of the task it was in the middle of. Verified by
    rendering through the model's own template: reasoning visible as stored,
    gone as sent. It undid §3.3 — the model re-decided fixes it had already
    made — while the stored history looked perfectly intact.

    A tool result is not a query to any template, so the note goes on the end
    of it — as a copy; the stored message is never touched. At the start of a
    task, where the last message is the user's own, there is no reasoning yet
    to lose and a separate message is harmless.
    """
    last = messages[-1] if messages else None
    if last is not None and last.get("role") == "tool":
        body = str(last.get("content") or "")
        return messages[:-1] + [dict(last, content=f"{body}\n\n{text}")]
    return messages + [{"role": "user", "content": text}]


def _with_tail(messages: list[dict], config: dict, tracker,
               turns_left: int | None = None, state_ref=None) -> list[dict]:
    """Everything appended to the request but never stored: map, then budget.

    Kept together because both must sit at the tail for cache reasons and both
    must be skipped while a tool call is outstanding.
    """
    out = _with_focus_map(messages, config, tracker)

    # What the session has established, rebuilt each turn from the transcript.
    #
    # Elision takes the content of a tool result and, without this, the fact of
    # it too — so the model loses not just a file's bytes but any record that it
    # ever looked. Measured on a build task: one file read nine times in forty
    # turns. The bulk is safe to lose because it can be fetched again; the
    # structure is small and cannot.
    if out and out[-1].get("role") != "assistant":
        from . import context as _ctx2
        if (ws := _ctx2.working_set(
                messages, ledger=getattr(state_ref, "ledger", None))):
            out = _extend_tail(out, messages, ws)

    if turns_left is not None and out and out[-1].get("role") != "assistant":
        # Tell the model how much room it has left.
        #
        # It was given a hard cap of tool rounds and no way to see it, so it
        # explored at the same rate whether it had thirty rounds left or two,
        # and six of eleven benchmark tasks simply stopped mid-investigation.
        # A person told "you have five minutes" prioritises differently; hiding
        # the budget and then failing the run for exceeding it is the harness
        # working against the model rather than with it.
        note = (f"[{turns_left} tool round(s) remain for this task. "
                f"Prioritise finishing over exploring further.]")
        out = _extend_tail(out, messages, note)
    return out


def _extend_tail(out: list[dict], messages: list[dict], text: str) -> list[dict]:
    """Add `text` to the request's tail: to what the harness already appended
    this request if anything, otherwise via _attach."""
    if out is messages:
        return _attach(messages, text)
    last = out[-1]
    return out[:-1] + [dict(last, content=f"{last.get('content') or ''}\n\n{text}")]


def _with_focus_map(messages: list[dict], config: dict, tracker) -> list[dict]:
    """Append a freshly-built, task-focused repo map to the request.

    Returned as a new list — the map is never written into ``state.messages``,
    so it does not accumulate across turns and never goes stale. It goes last
    for cache reasons: content before it stays cached, and content after it is
    new this turn regardless (§5.1).

    Skipped while a tool call is outstanding, because a user message between an
    assistant's tool_calls and their tool responses is an alternation violation.
    """
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
        return messages
    from . import context as _ctx
    body = _ctx.focus_map(config, messages, tracker)
    if not body:
        return messages
    return _attach(messages, "[repository map — generated now for this request, "
                             "reflects the current state of the files]\n\n" + body)


def _compaction_due(state: State, config: dict, schemas: list[dict] | None = None) -> int:
    """The history's size if a compaction should run now, else 0.

    Separate from the compaction itself so the caller can say one is starting
    before the tens of seconds it takes.
    """
    from . import config as _cfg
    from . import context as _ctx
    if not config.get("llama_ctx"):
        return 0                       # window unknown: let the server decide
    size = _ctx.estimate_tokens(state.messages, state.system)
    if size <= _cfg.history_budget(config, schemas):
        return 0
    # Do not re-run a compaction that already could not reach the budget: see
    # the note in _compact_if_needed. Wait until the history has genuinely
    # grown past the level achieved last time.
    if state.compact_floor and size < state.compact_floor * 1.15:
        return 0
    return size


def _compact_if_needed(state: State, config: dict, schemas: list[dict] | None = None) -> int:
    """Keep the conversation inside the window. Returns tokens reclaimed.

    The history is not the whole request. Every request also carries the tool
    schemas, the per-turn focus map, and a reservation for the reply the server
    is about to generate — none of which compaction can shrink. Budgeting the
    messages against the window alone therefore overflows by exactly the size of
    everything else, and the server answers with a 400 that kills the run
    mid-turn:

        request (17056 tokens) exceeds the available context size (16384)

    Three such crashes in one 12-task benchmark. So the history is budgeted
    against the room that is actually left for it, and `compact_at` becomes what
    its name suggests — a safety fraction of that room, not of the raw window.
    Compacting early costs some recent tool output; crossing the window is
    unrecoverable.
    """
    from . import config as _cfg
    from . import context as _ctx
    if not (size := _compaction_due(state, config, schemas)):
        return 0
    budget = _cfg.history_budget(config, schemas)

    # Do not re-run a compaction that has already been tried and could not
    # reach the budget.
    #
    # A summary is written by the model and cannot be expected to land on an
    # exact number, and some histories are simply not reducible any further —
    # the recent turns are kept verbatim because the model is mid-thought in
    # them. When that happened the history stayed above the trigger, so
    # compaction fired again on the very next turn: measured, 15 compactions in
    # 45 turns, each costing a generation and each invalidating the prefix
    # cache. Close enough is the correct outcome; retrying it is not.
    #
    # So remember the level actually achieved and wait until the history has
    # genuinely grown past it before trying again.

    # One mechanism: the model writes a note and the old exchanges are replaced
    # by it. Nothing is elided — there are no markers standing in for content
    # the model can no longer read.
    target = int(budget * float(config.get("compact_to", 0.6)))
    state.messages, freed = _ctx.compact_with_model(
        state.messages, target, config["model"], state.system, config,
        thinking=state.recent_thinking)
    state.compact_floor = _remember_floor(state, budget, _ctx)
    return freed


def _remember_floor(state, budget: int, _ctx) -> int:
    """The size compaction actually reached, when it could not reach budget.

    Zero once it fits, so a history that becomes reducible again is compacted
    normally rather than being held above the trigger forever.
    """
    reached = _ctx.estimate_tokens(state.messages, state.system)
    return reached if reached > budget else 0


_RESUME_RE = re.compile(r"resume=(\d+)")


def _resume_point(messages: list[dict], call_id_by_sig: dict, sig) -> int:
    """Where a part-served result for ``sig`` left off, per the transcript.

    Read back out of the message rather than tracked alongside it: a counter
    beside the conversation drifts the moment the exact bytes sent differ from
    what was counted, and it outlives the compaction that removed the piece it
    refers to. Here it vanishes with the message, and starting over is then the
    right answer.
    """
    cid = call_id_by_sig.get(sig)
    if cid is None:
        return 0
    for m in reversed(messages):
        if m.get("role") == "tool" and m.get("tool_call_id") == cid:
            hits = _RESUME_RE.findall(str(m.get("content") or ""))
            return int(hits[-1]) if hits else 0
    return 0


def repair_history(messages: list[dict]) -> int:
    """Give every assistant tool_call a matching tool response. Returns how many
    stubs were added.

    Needed after an interrupt. The loop appends one tool message per call, so
    Ctrl-C between two parallel calls leaves the first answered and the second
    dangling — and an assistant tool_call with no matching tool response is a
    guaranteed 400 on the very next request. Popping the assistant message only
    fixes the case where nothing was dispatched yet; stubbing the gaps works in
    both, and keeps whatever the model had already said.
    """
    last_asst = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
            last_asst = i
            break
    if last_asst is None:
        return 0

    wanted = [tc["id"] for tc in messages[last_asst]["tool_calls"]]
    answered = {m.get("tool_call_id") for m in messages[last_asst + 1:]
                if m.get("role") == "tool"}
    missing = [tid for tid in wanted if tid not in answered]
    for tid in missing:
        messages.append({"role": "tool", "tool_call_id": tid,
                         "content": "[interrupted by user before this ran]"})
    return len(missing)


def _checkpoint_label(name: str, params: dict) -> str:
    """What the checkpoint list will say this change was."""
    if name == "Bash":
        return f"Bash: {params.get('command', '')}"
    return f"{name} {params.get('file_path', '')}"


def run(
    state: State,
    config: dict,
    ask_permission: Callable[[str, dict], bool] | None = None,
    tracker: Any = None,
) -> Generator[Any, None, None]:
    """Run the agent until it stops calling tools. Yields events."""
    schemas = tools.schemas_for(config)
    max_turns = int(config.get("max_turns", 100))
    tools.new_turn()
    _count_exactly(config)
    config["_stops_this_turn"] = 0
    stuck_told = False
    alt_from: tuple[str, str] | None = None     # (test command, its last output)
    state.continuations = 0
    state.empty_retries = 0

    for used_turns in range(max_turns):
        # Compact *inside* the loop, not only between user turns.
        #
        # Compaction used to live solely in the REPL, between turns. But one
        # turn can run dozens of tool calls, each appending its output, so a
        # single request could exceed the window and hard-fail. Measured on a
        # benchmark task: 21,352 tokens against a 16,384-token window, returned
        # as a 400 that killed the run. Nothing had checked the budget since the
        # turn began.
        if (due := _compaction_due(state, config, schemas)):
            yield Compacting(due)
            t0 = _time.monotonic()
            if (freed := _compact_if_needed(state, config, schemas)):
                yield Notice(f"compacted: reclaimed ~{freed:,} tokens "
                             f"in {_time.monotonic() - t0:.0f}s")

        turn: AssistantTurn | None = None
        # Warn only near the end: a budget mentioned every turn is noise, and
        # it would also change the request tail on every single turn.
        left = max_turns - used_turns
        request = _with_tail(state.messages, config, tracker,
                             left if left <= max(3, max_turns // 5) else None,
                             state_ref=state)
        for event in stream_complete(config["model"], state.system, request, schemas, config):
            if isinstance(event, (TextChunk, ThinkChunk, ToolDraft)):
                yield event
            elif isinstance(event, StoppedCircling):
                yield Notice(f"stopped deliberating — {event.reason}")
                n = int(config.get("_stops_this_turn", 0))
                if n in (2, 3):
                    yield Notice("hint to the model: " + (
                        "say what the check expects and what the code does instead"
                        if n == 2 else "look up how this is normally done "
                        "(WebSearch, then WebFetch)"))
                if n >= 4 and not stuck_told:
                    stuck_told = True
                    yield Notice("the model looks stuck — its reasoning has gone "
                                 "round in circles four times on this task. Ctrl-C "
                                 "and give it a hint if you know what is wrong.")
            elif isinstance(event, Continuing):
                # Reported while it happens, not after. A turn that has not
                # ended is exactly the one whose cost is invisible.
                yield Notice(f"still writing — continuation {event.round}/"
                             f"{event.rounds}, {event.elapsed:.0f}s so far")
            else:
                turn = event
        assert turn is not None

        # Surface continuation. It is the right behaviour (DESIGN.md 5.3g) but
        # it costs a full generation per round, so it must never be invisible:
        # a turn that quietly continues eight times looks like a hung harness.
        if turn.continuations:
            yield Notice(f"continued {turn.continuations}x past the reply cap")

        if (thought := (turn.thinking or "").strip()):
            state.recent_thinking.append(thought)
            del state.recent_thinking[:-6]        # keep the last few turns

        msg = turn.to_message()
        state.messages.append(msg)
        valid = _strip_malformed(turn, msg)
        _save(state, config)

        if hint := _handle_truncation(turn, msg, state, config):
            yield Notice("output truncated — continuing")
            state.add_user(hint)
            continue

        if not valid:
            # An empty turn is not a completion — nudge before believing it.
            if hint := _handle_empty_turn(turn, msg, state, config):
                yield Notice("empty response — prompting the model to act")
                state.add_user(hint)
                continue
            yield TurnDone(turn)
            return

        # Bookmarks outlive the content they point into, so pagination can keep
        # advancing through a file across compaction.
        from . import context as _ctx
        all_ids = _ctx.tool_call_ids(state.messages)

        # Dispatch every tool call, appending one tool message per call. The
        # count and order must match the assistant's tool_calls exactly.
        for tc in valid:
            name, params = tc["name"], tc["input"]
            yield ToolStart(name, params, used_turns + 1, max_turns)

            denied, feedback = False, ""
            if name in tools.MUTATING and "_stopped" not in params:
                # Before the call, not after: "the tree before the agent
                # touched anything" cannot be reconstructed once it has.
                checkpoint.ensure_baseline(config, state.session_id)
                # Read live, not once per turn: answering "always" in the
                # middle of a turn has to stop the prompts for the rest of it.
                if (not (config.get("accept_all") or config.get("_accept_session"))
                        and ask_permission is not None):
                    # True allows. Anything else denies — and a string is the
                    # user saying what to do instead, which is worth more to
                    # the model than a bare refusal it has to guess around.
                    verdict = ask_permission(name, params)
                    if verdict is not True:
                        denied = True
                        feedback = verdict.strip() if isinstance(verdict, str) else ""
            if denied:
                result = ("The user denied this action. Do not retry it. "
                          + (f"They said: {feedback}" if feedback
                             else "Ask what to do instead."))
            elif (resume := _resume_point(state.messages, all_ids,
                                          sig := (name, json.dumps(params, sort_keys=True)))):
                # The same read-only call, and last time it did not all fit.
                # Asking again is the model saying "the rest, please" — which is
                # the only sensible move it has — so give it the next piece
                # instead of refusing.
                result = tools.dispatch(name, params, config, tracker,
                                        continue_from=resume)
            else:
                # Every call runs. A repeat is the model asking again, and the
                # answer to that is the answer, not a lecture about having
                # asked before.
                result = tools.dispatch(name, params, config, tracker)

            yield ToolEnd(name, result, denied)
            if tools.struggle_level() >= 3 and not stuck_told:
                # The last hint is for the person, not only the model: this is
                # where a sentence of human knowledge is worth more than more
                # attempts.
                stuck_told = True
                if name == "Bash":
                    alt_from = (params.get("command", ""), result)
                yield Notice(f"the model looks stuck — the same failure keeps "
                             f"coming back: {tools._FAIL_STREAK['sig']}. "
                             f"Ctrl-C and give it a hint if you know what is wrong.")
            if (name in tools.MUTATING and not denied
                    and not result.startswith("Error")):
                # A checkpoint per accepted change, in a shadow repository —
                # the user's own git history is never touched. Undoing a wrecked
                # file stops depending on the agent having been polite about it.
                if made := checkpoint.snapshot(config, state.session_id,
                                               _checkpoint_label(name, params)):
                    if made[1]:
                        yield Notice(made[1])
            state.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
            # Record it before compaction can ever see it. Only the head of the
            # result is kept: working_set reads the first line for a command's
            # outcome and tests the "Error" prefix, and nothing else.
            state.ledger.append((tc["id"], name, params, result[:200]))
            # Saved after every result, not at the end of the turn: a crash,
            # a closed terminal or a sleeping laptop mid-turn used to lose
            # everything since the user's last message.
            _save(state, config)

        if alt_from:
            # Once every result of this round is in, so the history stays
            # well-formed: try several fixes and keep the one the tests prefer.
            from . import alternatives
            cmd, out = alt_from
            alt_from = None
            yield from alternatives.run(state, config, tracker, cmd, out, schemas)
            _save(state, config)

    _save(state, config)
    yield Notice(f"stopped after {max_turns} tool rounds")


def _count_exactly(config: dict) -> None:
    """Use the server's tokenizer for every size check, if it has one.

    Once per run of the loop. The budget used to rest on a character estimate
    corrected by a calibration factor — which compounded to four times the real
    size and made the harness compact at a quarter of its budget. Counting is
    what the server is for; the estimate stays only for servers that cannot.
    """
    if config.get("_counter_checked"):
        return
    config["_counter_checked"] = True
    from .provider import split_model
    if split_model(config.get("model", ""))[0] != "local":
        return
    from . import server
    from . import context as _ctx
    if (counter := server.exact_counter(config)):
        _ctx.use_tokenizer(*counter)


def _save(state: State, config: dict) -> None:
    from . import session
    session.save_point(state, config.get("_cwd", ""))
