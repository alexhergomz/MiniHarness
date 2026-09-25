"""When the model is stuck, try several fixes and keep the one the tests prefer.

Watched on a live 9B build: the model stated its bug exactly — "the children
only include the Value, not the scalar" — searched, read a tutorial, and still
spent half an hour moving between 13 and 15 of 18 tests passing. Its attempts
differed a lot from one to the next; the right design was plausibly one sample
away, and the one judge that settles it, the failing test, costs nothing to
run.

That is the whole of this module. At the point the harness already knows the
model is stuck (tools.struggle_level() == 3), and only when checkpoints make it
safe to try and undo:

  1. ask the model for its fix, several times, from the same point;
  2. for each, rewind to that point, apply just its Write/Edit calls, and run
     the command that kept failing;
  3. keep the best, if it beats where things stood — otherwise undo them all.

It adds samples; it does not steer them. Each request is the model's own
conversation with one instruction attached to its last tool result (never a
new user message, so its reasoning is kept — §3.3.1), and the winning reply
enters the history as the model's own turn, with a note saying what was tried.
The repeated-sampling literature is the reason to expect this to work: with an
objective verifier, the best of several attempts beats a single one by a wide
margin (AlphaCode; "Large Language Monkeys", Brown et al. 2024; Agentless).
"""

from __future__ import annotations

import copy
import re

from . import checkpoint, tools

ALT_PROMPT = (
    "[The same failure has come back many times, so the harness is asking for "
    "your fix from here more than once and will keep whichever one the tests "
    "like best. Reply with the Write or Edit calls that fix the failure — the "
    "complete change, in this one reply. Other tool calls are ignored here.]"
)

_EXIT = re.compile(r"\[exit (\d+)\]")


def _score(out: str) -> tuple[int, int]:
    """(failed, passed) from a test run's output. Lower failed wins."""
    counts = None
    tail = "\n".join(l for l in out[-40000:].splitlines() if len(l) <= 500)
    for m in tools._SUITE_RE.finditer(tail):
        if m.group("passed") or m.group("failed"):
            counts = m
    if counts:
        failed = int(counts.group("failed") or 0) + int(counts.group("error") or 0)
        return failed, int(counts.group("passed") or 0)
    return (1 if _EXIT.search(out) else 0), 0


def _better(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[0] or (a[0] == b[0] and a[1] > b[1])


def _fmt(s: tuple[int, int]) -> str:
    return f"{s[0]} failed, {s[1]} passed"


def _head(config: dict, sid: str) -> str:
    """A checkpoint of the tree as it is now."""
    made = checkpoint.snapshot(config, sid, "before trying alternative fixes")
    if made:
        return made[0]
    rows = checkpoint.history(config, sid, limit=1)
    return rows[0][0] if rows else ""


def run(state, config: dict, tracker, test_cmd: str, last_out: str, schemas: list):
    """Try alternatives from the current point. A generator of loop events."""
    from . import loop as _loop
    from .loop import Notice, _attach
    from .provider import AssistantTurn

    n = int(config.get("best_of", 3) or 0)
    sid = state.session_id
    if n < 2 or not sid or not checkpoint.enabled(config):
        return
    base = _head(config, sid)
    if not base:
        return
    baseline = _score(last_out)
    yield Notice(f"trying {n} alternative fixes and keeping whichever the tests "
                 f"like best — the same failure has come back seven times")

    request = _attach(state.messages, ALT_PROMPT)
    candidates = []
    for i in range(1, n + 1):
        checkpoint.restore(config, sid, base)
        turn = None
        # Through the loop's own model call, so an alternative is asked for
        # exactly the way every other turn is — same guards, same stops.
        for ev in _loop.stream_complete(config["model"], state.system, request,
                                        schemas, config):
            if isinstance(ev, AssistantTurn):
                turn = ev
        calls = [tc for tc in (turn.tool_calls if turn else [])
                 if tc["name"] in ("Write", "Edit")
                 and not {"_raw", "_stopped"} & set(tc.get("input", {}))]
        if not calls:
            yield Notice(f"alternative {i}: no edit proposed")
            continue
        results = [tools.dispatch(tc["name"], tc["input"], config, tracker) for tc in calls]
        out = tools.dispatch("Bash", {"command": test_cmd}, config, None)
        score = _score(out)
        made = checkpoint.snapshot(config, sid, f"alternative fix {i}")
        candidates.append((score, i, made[0] if made else base, turn, calls, results, out))
        yield Notice(f"alternative {i}: {_fmt(score)}")

    best = min(candidates, key=lambda c: (c[0][0], -c[0][1])) if candidates else None
    tried = ", ".join(f"#{c[1]} {_fmt(c[0])}" for c in candidates) or "none produced an edit"
    if best is None or not _better(best[0], baseline):
        checkpoint.restore(config, sid, base)
        _note_last(state, f"\n[the harness tried {n} alternative fixes from here ({tried}); "
                          f"none did better than the current code ({_fmt(baseline)}), so "
                          f"nothing was changed]")
        yield Notice(f"no alternative beat the current code ({_fmt(baseline)}) — "
                     f"kept it unchanged")
        return

    score, idx, sha, turn, calls, results, out = best
    checkpoint.restore(config, sid, sha)
    chosen = copy.copy(turn)
    chosen.tool_calls = calls
    state.messages.append(chosen.to_message())
    for tc, res in zip(calls, results):
        state.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
        state.ledger.append((tc["id"], tc["name"], tc["input"], res[:200]))
    first = [l for l in out.splitlines() if l.strip()][-1:] or [""]
    _note_last(state, f"\n[the harness tried {n} alternative fixes from here ({tried}) and "
                      f"kept #{idx}: {_fmt(baseline)} → {_fmt(score)}. Running "
                      f"`{test_cmd}` now gives: {first[0].strip()[:200]}]")
    yield Notice(f"kept alternative {idx}: {_fmt(baseline)} → {_fmt(score)}")


def _note_last(state, text: str) -> None:
    for m in reversed(state.messages):
        if m.get("role") == "tool":
            m["content"] = f"{m.get('content') or ''}{text}"
            return
