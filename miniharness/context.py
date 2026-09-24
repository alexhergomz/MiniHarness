"""Context management: the stable prefix, the file tracker, and compaction.

The governing constraint is prefix stability (DESIGN.md §5.1). llama.cpp caches
the KV of a matching prompt prefix, so if the first N tokens are byte-identical
to last turn, prefill is free — and on an 8 GB card prefilling 30 k tokens costs
seconds. Everything here is arranged so the head of the prompt never moves:

  * the system prompt is assembled in a fixed order with nothing volatile in it
    (no timestamps, no session ids, no "you are running at 14:32");
  * the repo map is regenerated only when the tree's mtime fingerprint changes,
    and that is a deliberate, logged invalidation;
  * compaction rewrites the *middle* of the conversation and never the head.

Compaction is deterministic — no summarizer model call. Old tool outputs are the
bulk of a long agent conversation and the least valuable per token, so they get
replaced by a note the model wrote. Only if that isn't enough do whole exchanges get
dropped. A summarization pass would cost a full generation and, worse, produce a
different prefix every time it ran.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

BASE_PROMPT = """You are a coding agent working in a terminal on the user's machine.

Tools: a map of the repository is provided with each request, so start from it \
rather than exploring blindly. Grep finds things by text and also finds \
definitions — searching for a name reports near-misses if it is spelled \
differently. Read a file to see it; reading a large one returns its definitions \
and line numbers, so read a specific part with offset. Use Bash to run tests, \
git, and builds. WebSearch when an error or API is unfamiliar.

Rules:
- Read a file before you edit it. Prefer Edit over Write for existing files.
- Make the smallest change that solves the problem. Do not add dependencies, \
abstractions, or error handling that was not asked for.
- Verify your work: run the tests or the code before reporting success.
- If something fails, say so plainly with the output. Do not claim a fix you \
have not run.
- Match the style, naming, and comment density of the surrounding code.
- A search that finds nothing is an answer. Say what you looked for and that \
it is not there, then move on — do not run the same search again in another \
form.
- Be concise. Explain what you did, not what you are about to do."""


class FileTracker:
    """Remembers which files were read, so Write/Edit can require a prior Read."""

    def __init__(self):
        self._read: set[str] = set()

    @classmethod
    def from_messages(cls, messages: list[dict], cwd: str | None = None) -> "FileTracker":
        """Rebuild from a transcript, so a resumed session knows what it read.

        The tracker lives in memory and starts empty, so after /resume a file
        read last session counted as unread and the read-before-edit rule fired
        on a file whose contents were sitting in the model's context. Measured:
        the first Edit after a resume is refused, costing a turn to re-read
        something already visible.

        Derived from the transcript rather than persisted alongside it — a saved
        tracker would drift from the messages it describes.
        """
        import json as _json
        import os as _os

        tr = cls()
        pending: dict[str, str] = {}
        for m in messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    if fn.get("name") != "Read":
                        continue
                    try:
                        args = _json.loads(fn.get("arguments") or "{}")
                    except (ValueError, TypeError):
                        continue
                    if path := args.get("file_path"):
                        pending[tc.get("id", "")] = str(path)
            elif m.get("role") == "tool":
                path = pending.pop(m.get("tool_call_id", ""), None)
                body = str(m.get("content") or "")
                # Only a read that actually returned something counts.
                if path and not body.startswith("Error"):
                    p = _os.path.expanduser(path)
                    if not _os.path.isabs(p):
                        p = _os.path.join(cwd or _os.getcwd(), p)
                    tr.mark_read(p)
        return tr

    def mark_read(self, path: str) -> None:
        self._read.add(os.path.abspath(path))

    def has_read(self, path: str) -> bool:
        return os.path.abspath(path) in self._read

    def forget(self, path: str) -> None:
        self._read.discard(os.path.abspath(path))


# ── The stable prefix ───────────────────────────────────────────────────────
_MAP_CACHE: dict[str, tuple[str, str]] = {}  # root -> (fingerprint, rendered map)


def fingerprint(root: str) -> str:
    """Cheap mtime fingerprint of a source tree. Changes iff a file changed."""
    from .repomap import find_src_files
    latest, count = 0.0, 0
    for f in find_src_files(root):
        try:
            latest = max(latest, os.path.getmtime(f))
            count += 1
        except OSError:
            continue
    return f"{count}:{latest:.0f}"


def build_system(config: dict, root: str | None = None) -> str:
    """Assemble the system prompt. Byte-stable unless the repo actually changed.

    Order is fixed and must stay fixed: base prompt, then project instructions,
    then repo map. Appending at the end is the only safe way to extend this.
    """
    root = root or config.get("_cwd") or os.getcwd()
    parts = [BASE_PROMPT]

    # Project-specific instructions. The ratchet rule applies: every line in
    # here should be traceable to a specific thing that went wrong.
    for name in ("AGENTS.md", "CLAUDE.md", ".miniharness.md"):
        p = Path(root) / name
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if text:
                parts.append(f"# Project instructions ({name})\n{text}")
            break

    # The repo map is deliberately NOT here. It used to be, which made the
    # system prompt vary per repo and go stale the moment the agent edited a
    # file. It is now injected fresh at the tail of each request — see
    # focus_map(). Keeping it out makes this prompt genuinely constant, which is
    # the strongest possible form of prefix stability (§5.1).
    return "\n\n".join(parts)


# ── The focus map: fresh, relevant, and ephemeral ───────────────────────────
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_PATH_RE = re.compile(r"[\w./-]+\.[A-Za-z]{1,4}\b")
# Words that appear in every request and would only dilute the ranking.
_STOPWORDS = frozenset("""
the and for with that this from have has was were are you your they them then
than into out get set add fix bug code file files function class method test
tests please make sure run use using need should would could there where when
what which why how all any not but its it's use used new old
""".split())


def mentions_from(messages: list[dict], limit: int = 3) -> set[str]:
    """Identifiers and filenames worth steering the map toward.

    Drawn from the user's request and the model's own prose — deliberately not
    from tool results, which are file contents and would swamp the signal with
    every identifier in the repo.
    """
    out: set[str] = set()
    considered = 0
    for m in reversed(messages):
        if m.get("role") not in ("user", "assistant"):
            continue
        text = str(m.get("content") or "")
        if not text:
            continue
        out |= {w for w in _IDENT_RE.findall(text)
                if w.lower() not in _STOPWORDS and not w.isdigit()}
        out |= set(_PATH_RE.findall(text))
        considered += 1
        if considered >= limit:
            break
    return out


# Progressive compaction — the model maintaining NOTES.md as its own memory —
# was implemented and measured against plain eviction on a two-turn task where
# a constraint given in turn 2 had to survive. It lost on every axis: 2-4x the
# wall time, 2-3x the tool calls, and it FAILED the run that eviction passed.
#
# The mechanism is self-defeating. Checkpointing costs generations; those turns
# grow the context; the extra context forces more eviction — so it accelerated
# the pressure it existed to relieve, and evicted the very instruction it was
# meant to preserve. Plain eviction never lost it, because tool output was cut
# before any user message is dropped.
#
# Worth revisiting only for horizons long enough that eviction destroys
# something needed, which this task never reached.

def focus_map(config: dict, messages: list[dict], tracker=None) -> str | None:
    """A small map of *this task*, rebuilt for the current turn.

    Three properties the old static map could not have at once:

    * **Fresh** — rebuilt per turn against current mtimes, so it cannot be stale
      and needs no staleness disclaimer.
    * **Relevant** — PageRank personalized toward the files already open and the
      identifiers in the request, so it spends its budget on the task instead of
      on whatever the repo's most-referenced utility happens to be.
    * **Ephemeral** — injected at the tail of the request and never stored in
      history, so it costs its tokens once rather than accumulating.

    The cache cost is bounded: because it sits at the tail, changing it only
    invalidates from its own position, and everything after it is new that turn
    anyway. Per-turn extra prefill is roughly the map's own size.
    """
    if not config.get("repo_map", True):
        return None
    from . import config as _cfg
    from .repomap import HAVE_GRAPH, repo_map
    if not HAVE_GRAPH:
        return None

    root = config.get("_cwd") or os.getcwd()
    focus = []
    if tracker is not None:
        for p in getattr(tracker, "_read", ()):
            try:
                focus.append(os.path.relpath(p, root))
            except ValueError:
                continue
    try:
        return repo_map(root, focus=focus[:20],
                        max_tokens=_cfg.budget(config, "repo_map_share", floor=200),
                        mentions=mentions_from(messages))
    except Exception:
        return None


# Kept only so an old session or test referring to it still imports. The map is
# now rebuilt every turn (see focus_map), so there is nothing to disclaim.
MAP_STALENESS_NOTE = ""


# ── Compaction ──────────────────────────────────────────────────────────────

_RESUME = re.compile(r"resume=(\d+)")

# Output that can simply be asked for again. Bash is deliberately absent:
# re-running a command is not a way to recover its output — it may be slow,
# have side effects, or not reproduce — so its results are given up last and
# never advertised as recoverable.
_RECOVERABLE = frozenset({"Read", "Grep", "Glob"})


_DIGITS = re.compile(r"\d+")


_TARGETED = {"Read": "file_path", "Edit": "file_path", "Write": "file_path"}
_MUTATES = {"Edit", "Write", "Bash"}


def _tool_calls_by_id(messages: list[dict]) -> dict[str, tuple[str, dict, int]]:
    """tool_call_id -> (tool name, arguments, position in the conversation)."""
    out: dict[str, tuple[str, dict, int]] = {}
    for i, m in enumerate(messages):
        for tc in m.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except (json.JSONDecodeError, KeyError, TypeError):
                args = {}
            out[tc.get("id", "")] = (tc.get("function", {}).get("name", ""), args, i)
    return out


def raw_chars(messages: list[dict], system: str = "") -> int:
    """Characters that will actually reach the model.

    Reasoning counts, but only where the template will render it. Measured
    against this server: within an open assistant turn every step's
    ``reasoning_content`` is rendered (three steps cost 225 prompt tokens
    against 173 without), while a turn the model has already finished is
    dropped entirely — identical prompt tokens with and without it.

    A turn is open from the last user message onward. Counting earlier turns
    too would be safe but wasteful; counting none of it is the §1 mistake, a
    budget measured against the convenient part rather than what is sent.
    """
    n = len(system)
    last_user = max((i for i, m in enumerate(messages)
                     if m.get("role") == "user"), default=-1)
    for i, m in enumerate(messages):
        n += len(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            n += len(str(tc.get("function", {}).get("arguments", "")))
        if i > last_user:
            n += len(str(m.get("reasoning_content") or ""))
    return n


def estimate_tokens(messages: list[dict], system: str = "") -> int:
    """Estimated tokens, corrected by what the server has actually reported.

    "4 chars ~= 1 token" is close for prose and code and badly wrong for the
    output agents handle most. Measured against the real tokenizer:

        prose 0.89x   python 1.00x   schemas 1.07x   traceback 1.27x
        JSON 1.55x    ls -l 2.35x    hexdump 3.55x

    So a budget of 9,152 estimated tokens can be 21,500 real ones, and every
    budget in the harness sits on top of it. This was the root cause of the 400s
    that survived three separate rounds of window-arithmetic fixes: the
    arithmetic was right and the ruler was wrong.

    No fixed divisor fixes it — len/2.5 still undercounts a hexdump by 2.2x
    while wasting 1.8x on prose. But the server reports the true prompt length
    with every response, so the correction is measured rather than guessed, and
    adapts to the tokenizer and to whatever the session is actually handling.
    """
    return int(raw_chars(messages, system) / 4 * _calibration)


_calibration = 1.0


def calibrate(actual_prompt_tokens: int, estimated: int) -> float:
    """Learn the correction from one request the server has counted for us.

    Deliberately asymmetric: it jumps straight to any underestimate and decays
    back only slowly. Undercounting overflows the window and kills the run;
    overcounting compacts a little early and costs some recent tool output.
    """
    global _calibration
    # Below this the chat template's fixed markup dominates and the ratio says
    # more about overhead than about how the content tokenises.
    if actual_prompt_tokens > 0 and estimated > 1500:
        ratio = actual_prompt_tokens / estimated
        _calibration = min(4.0, max(1.0, _calibration * 0.97, _calibration * ratio))
    return _calibration


SUMMARY_SYSTEM = """You extract facts from a transcript of work already done.

You are not the agent and you do not continue the work. Your only job is to \
carry forward what was learned, so that whoever reads your notes never has to \
re-open a file or re-run a command to recover it.

Rules:
- Record concrete values, never categories. "cmake 3.22.1" not "cmake was \
installed". "line 4 of CMakeLists.txt" not "an early line".
- Copy identifiers, paths, flags, versions and error text exactly as they \
appear. A value you paraphrase is a value that has been lost.
- Quote the operative line of an error verbatim.
- Write only what is in the transcript. If you did not see it, do not write it \
— an invented version number is worse than a missing one, because it will be \
believed.
- No advice, no commentary, no restating the task."""

NOTE_PREFIX = "Notes from the earlier part of this session:\n"


def _is_note(m: dict) -> bool:
    return (m.get("role") == "assistant"
            and str(m.get("content") or "").startswith(NOTE_PREFIX))


# A required section with nothing to put in it is a request to invent.
#
# Observed: a run that had executed no commands at all produced a note reading
# "Exact Error: KeyError: 'max_file_lines'" — an error it could not have seen,
# for a line that did not exist. The note then became the transcript's ground
# truth and the model spent 27 greps hunting for it. The system prompt already
# forbids invention; that was not enough, because the *template* still demanded
# a filled slot. So every section that can legitimately be empty now names the
# words to write when it is.
SUMMARISE = (
    "Your context is about to be shortened and the tool output above will be "
    "removed. Write the notes your successor needs to finish this task without "
    "re-reading anything.\n\n"
    "Cover, in this order and nothing else:\n"
    "1. What you have established — the specific facts you learned from files "
    "and commands (names, line numbers, flags, values), not that you looked. "
    "Include what you worked out, not only what you saw: the diagnosis, the "
    "cause, the plan you formed. For each file you read, say which lines you "
    "read and what was in them.\n"
    "   Record what a search or command established does not exist — "
    "'grep for X in tests/ returns nothing, so it is not there yet' is a fact "
    "and it stops the search being repeated. Never record that you have not "
    "yet looked at something, or that a file's contents were not visible to "
    "you: that will be believed, and the reading will simply be done again.\n"
    "2. What you have already changed. If you have changed nothing yet, write "
    "'nothing changed yet'.\n"
    "3. What is still wrong. Quote the error verbatim only if a command you "
    "actually ran produced it. If nothing has failed, or you have not run "
    "anything yet, write 'nothing has failed yet' — do not describe an error "
    "you have not seen.\n"
    "4. What you were about to do next.\n\n"
    "An empty section is a fact worth recording. Writing 'nothing has failed "
    "yet' is a correct answer; inventing something to fill the section is not.\n"
    "Be specific and terse. Anything you leave out is gone."
)


def summarise_span(messages: list[dict], model: str, system: str, config: dict,
                   thinking: list | None = None) -> str | None:
    """Ask the model what it has learned, before the evidence is discarded.

    Derived structure (see working_set) can say *what was looked at* — it cannot
    say what was found in it. "You read CMakeLists.txt three times" does not
    carry "ONLY_SIMPLE is defined at line 480 and needs minisat", and only the
    model knows that, and only while the content is still in front of it.

    Note this is not the progressive checkpointing that was measured and
    rejected: that ran on every step and its cost compounded — the turns it
    spent grew the context, which forced more eviction. This runs once when the
    budget is actually hit, which after the hysteresis fix is a handful of times
    per task, and it is weighed against the re-reads it prevents (nine of one
    file inside forty turns, measured).

    Runs greedy and without deliberation. Summarising is extraction, not
    generation: there is nothing here to be creative about, and sampling a line
    number or a flag name is pure downside. Both settings are scoped to this
    call — applying them to the agent's own turns changes the thing under test.

    Greedy decoding makes this model repeat itself, and here it did: on a 39k
    transcript it emitted one line about the fix 42 times in a row, ran to the
    cap and returned a note that was 13% distinct lines. That is where the cost
    of a compaction actually goes — measured on the same call, same prompt:

        temperature 0.0                     2048 tok, 101 s, hit the cap
        temperature 0.0 + frequency 0.4      290 tok,  14 s, stopped on its own
        temperature 0.3                     2048 tok, 101 s, hit the cap

    So the obvious fix is the wrong one: raising the temperature only makes the
    rambling non-identical, and it costs the determinism that keeps a line
    number from being resampled. A frequency penalty keeps greedy decoding and
    stops the loop. It is small on purpose — a summary must be free to repeat
    `max_file_lines` as often as the facts require.

    Returns None on any failure: a summary that did not arrive must degrade to
    dropping the oldest exchanges outright (see compact_with_model), never to a
    broken turn. It does not degrade to elision — there is no elision left, and
    a comment saying otherwise is an invitation to put it back.
    """
    from .provider import stream

    try:
        # The summariser gets its own system prompt. Handed the agent's, the
        # model stays in coding-agent voice and writes activity rather than
        # values — measured: it recorded "Installed cmake via apt-get" and
        # dropped "3.22.1", which the reader then confabulated as "3.28.2".
        ask = list(messages)
        if thinking:
            # What the model worked out, which the transcript never holds.
            recent = "\n\n".join(str(t) for t in thinking[-4:])[:6000]
            ask = ask + [{"role": "user", "content":
                          "Your own reasoning from these turns, for reference "
                          "while writing the notes:\n\n" + recent}]
        ask = ask + [{"role": "user", "content": SUMMARISE}]
        # The summariser is watched for circling, like every other path that
        # generates.
        #
        # It was the third one to be missed. The round loop was covered, then
        # _conclude, and this call — which goes straight to `stream` and never
        # passes through `stream_complete` — was not. Measured on a build task:
        # two notes ran to 27,606 characters, essentially the 7,864-token cap,
        # at 340s each. That was 683 of the 765 seconds the run spent
        # compacting, against 40s for the two notes that behaved.
        #
        # frequency_penalty does not catch it. It stops verbatim repetition,
        # and this is the paraphrasing kind — the same thing that defeated it
        # on the agent's own turns.
        from .provider import TextChunk, ThinkChunk, _Deliberation
        watch = _Deliberation(dict(config, think_limit=8192))   # a note, not a solution
        parts: list[str] = []
        turn = None
        # One attempt. This is a mechanical call with a fallback right behind
        # it, so retrying an unreachable model just spends the backoff before
        # doing what it was always going to do — and it is the agent's own turn
        # that pays. Measured: the suite went from 14s to 77s on retries alone.
        for ev in stream(model, SUMMARY_SYSTEM, ask, [],
                         dict(config, reply_share=0.06, disable_thinking=True,
                              temperature=0.0, frequency_penalty=0.4,
                              max_retries=0)):
            if type(ev).__name__ == "ToolDraft":
                continue                     # the summariser has no tools anyway
            if isinstance(ev, (TextChunk, ThinkChunk)):
                parts.append(ev.text)
                if watch.feed(ev.text):
                    break          # keep what it wrote; a partial note beats none
            else:
                turn = ev
        text = (getattr(turn, "text", "") or "").strip() or "".join(parts).strip()
        return text or None
    except Exception:
        return None


def working_set(messages: list[dict], limit: int = 1400,
                ledger: list | None = None) -> str | None:
    """What the model has established so far, as structure rather than bulk.

    Elision removes the *content* of a tool result and, until now, everything
    else with it. Measured on a build task: 33 of 40 results discarded, and the
    model read one CMakeLists.txt nine times and another five times inside forty
    turns — it had no record that it had ever read them, so re-reading was its
    only way to find out. Six of those forty turns did real work.

    The bulk is genuinely safe to lose: a file can be read again. The *structure*
    is not, and it is small. So it is appended to the request like the focus map,
    never stored.

    Derived from the transcript *and* from a ledger the loop appends to, because
    the transcript alone stops being sufficient the moment compaction runs.
    Measured: two files edited, then a compaction, and this function returned
    None — the edits lived in the replaced span, so the only record that any
    change had been made went with it. The note covering that span said "nothing
    changed yet", so the model had two independent sources agreeing it had done
    nothing, and re-derived its own work. One scenario compacted 13 times and
    spent 31 Reads against 3 Edits.

    The ledger is append-only and never shrinks: it is the one thing here that
    compaction must not be able to reach. Entries are merged by call id, so a
    resumed session — whose ledger starts empty — still derives what it can from
    the messages it loaded.

    Deliberately only facts the session actually produced: what was changed,
    how commands ended, what has been looked at and how often. A repeat count is
    the cheapest possible signal that the model is going in circles, and it is
    the one thing it cannot observe about itself.
    """
    calls = _tool_calls_by_id(messages)
    results: dict[str, str] = {}
    for m in messages:
        if m.get("role") == "tool":
            results[m.get("tool_call_id", "")] = str(m.get("content") or "")

    # Ledger entries come first: they are the older half of the session, the
    # part compaction has already taken out of `messages`. A call present in
    # both keeps its ledger position, so ordering stays chronological.
    merged: dict[str, tuple[str, dict, int]] = {}
    for pos, (cid, name, args, body) in enumerate(ledger or []):
        merged[cid] = (name, args, pos)
        results.setdefault(cid, body)
    base = len(merged)
    for cid, (name, args, pos) in calls.items():
        if cid not in merged:
            merged[cid] = (name, args, base + pos)
    calls = merged

    edits: list[str] = []
    commands: list[str] = []
    seen: dict[str, int] = {}
    order: list[str] = []

    for cid, (name, args, _pos) in sorted(calls.items(), key=lambda kv: kv[1][2]):
        body = results.get(cid, "")
        if name in ("Edit", "Write"):
            path = str(args.get("file_path") or "")
            if not body.startswith("Error"):
                what = str(args.get("old_string") or "")[:40].replace("\n", " ")
                edits.append(f"{path}{f' — replaced {what!r}' if what else ' — written'}")
        elif name == "Bash":
            cmd = " ".join(str(args.get("command") or "").split())[:70]
            tail = [ln for ln in body.strip().splitlines() if ln.strip()][-1:]
            outcome = (tail[0][:90] if tail else "(no output)")
            commands.append(f"{cmd} -> {outcome}")
        else:
            target = str(args.get("file_path") or args.get("pattern") or "")
            if not target:
                continue
            key = f"{name} {target}"
            if key not in seen:
                order.append(key)
            seen[key] = seen.get(key, 0) + 1

    lines: list[str] = []
    if edits:
        lines.append("Changed so far:")
        lines += [f"  - {e}" for e in edits[-8:]]
    if commands:
        lines.append("Commands run (most recent last):")
        lines += [f"  - {c}" for c in commands[-6:]]
    if order:
        lines.append("Already looked at ("
                     "repeat the call to see it again):")
        for key in order[-12:]:
            n = seen[key]
            lines.append(f"  - {key}" + (f"  [{n}x]" if n > 1 else ""))
    repeats = [k for k in order if seen[k] >= 3]
    if repeats:
        lines.append(f"You have fetched {repeats[0]} {seen[repeats[0]]} times. "
                     f"If you still need something from it, say what — otherwise "
                     f"act on what you have.")
    if not lines:
        return None
    out = "[what you have established so far — rebuilt from this session]\n" + "\n".join(lines)
    return out if len(out) <= limit * 4 else out[:limit * 4] + "\n  …"


def tool_call_ids(messages: list[dict]) -> dict:
    """Latest call id per read-only signature, regardless of whether its result
    still has a body. Used to find a pagination bookmark."""
    from .tools import MUTATING
    pending, ids = {}, {}
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if fn.get("name") in MUTATING:
                    ids.clear(); pending.clear()
                else:
                    pending[tc.get("id")] = (fn.get("name") or "",
                                             fn.get("arguments") or "")
        elif m.get("role") == "tool":
            sig = pending.pop(m.get("tool_call_id"), None)
            if sig:
                ids[sig] = m.get("tool_call_id")
    return ids


def _merge_adjacent_user_turns(messages: list[dict]) -> list[dict]:
    """Fold consecutive user messages into one.

    Rescuing user turns from eviction can leave them adjacent once the
    assistant/tool turns between them are gone, and user -> user is the
    alternation violation that makes models loop or repeat themselves — the same
    invariant the truncation fix protects.
    """
    out: list[dict] = []
    for m in messages:
        if (m.get("role") == "user" and out and out[-1].get("role") == "user"
                and not out[-1].get("tool_calls")):
            out[-1] = {"role": "user",
                       "content": f"{out[-1]['content']}\n\n{m['content']}"}
        else:
            out.append(m)
    return out


def compact_with_model(messages: list[dict], budget_tokens: int, model: str,
                       system: str, config: dict, keep_recent: int = 8,
                       thinking: list | None = None):
    """Replace the old part of the conversation with a note the model wrote.

    The whole mechanism, in order:

      user instructions   kept verbatim, always (governance)
      everything older    replaced by one note the model wrote about it
      recent exchanges    kept verbatim, bounded by a share of the budget

    Nothing is elided. A tool result is either still here in full or it is in
    the note; there are no markers standing in for content that used to exist.

    The tail is bounded by size rather than a message count because a count is
    a token quantity in disguise — measured on one compaction, eight recent
    messages were 88% of everything left afterwards, and with larger reads the
    same eight would be ten times that.
    """
    if estimate_tokens(messages) <= budget_tokens:
        return messages, 0
    before = estimate_tokens(messages)

    head_end = next((i for i, m in enumerate(messages) if m.get("role") != "user"), 0)

    # Keep as much recent history verbatim as the budget allows.
    #
    # A fixed count is a token quantity in disguise, and it cuts both ways: at
    # eight messages a 64k window discarded exactly as much as a 16k one, so
    # the extra room bought nothing. Walk back from the end while it still
    # fits, keeping at least one exchange.
    tail_cap = max(512, int(budget_tokens * 0.5))
    tail_start = len(messages)
    while tail_start > head_end:
        if (estimate_tokens(messages[tail_start - 1:]) > tail_cap
                and len(messages) - tail_start >= 2):
            break
        tail_start -= 1
    while tail_start < len(messages) and messages[tail_start].get("role") == "tool":
        tail_start += 1          # never start the tail on an orphan tool message
    if tail_start <= head_end:
        # Nothing older than the tail: a short conversation of very large
        # messages, where the tail *is* the conversation. Elision used to
        # shrink those in place. The equivalent without it is to summarise the
        # tail itself — the model still gets what was in the output, as prose
        # it wrote, rather than a marker saying the content is gone.
        tail_start = len(messages)

    span = messages[head_end:tail_start]
    note = summarise_span(span, model, system, config, thinking=thinking)
    if not note:
        # No note means nothing would carry the span forward, so the span stays
        # and the tail is trimmed instead. Losing the oldest exchanges outright
        # is worse than a note but better than a broken turn, and it is at
        # least honest about what is gone.
        out = _drop_oldest(messages, budget_tokens, head_end)
        return out, before - estimate_tokens(out)

    # Any earlier note was inside the span, so the new one already covers it.
    out = (messages[:head_end]
           + [m for m in span if m.get("role") == "user"]      # governance
           + [{"role": "assistant", "content": NOTE_PREFIX + note}]
           + messages[tail_start:])
    out = _merge_adjacent_user_turns(out)
    if estimate_tokens(out) > budget_tokens:
        out = _drop_oldest(out, budget_tokens, head_end)
    return out, before - estimate_tokens(out)


def _drop_oldest(messages: list[dict], budget_tokens: int,
                 head_end: int) -> list[dict]:
    """Last resort: drop whole exchanges from the front, keeping user turns.

    Never leaves a tool result whose call has gone, and never drops the note or
    a user instruction.
    """
    out = list(messages)
    i = head_end
    while estimate_tokens(out) > budget_tokens and i < len(out) - 2:
        m = out[i]
        if m.get("role") == "user" or _is_note(m):
            i += 1
            continue
        drop = {tc["id"] for tc in (m.get("tool_calls") or [])}
        out.pop(i)
        while i < len(out) and out[i].get("tool_call_id") in drop:
            out.pop(i)
    # Dropping the assistant/tool turns between two user messages leaves them
    # adjacent, and user -> user is the alternation violation that makes models
    # loop. Merge again after removing anything.
    return _merge_adjacent_user_turns(out)

