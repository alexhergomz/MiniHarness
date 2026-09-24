"""What a tool call will do — and did — rendered for the person watching it.

The permission prompt used to print a tool name and a path. That is not
oversight — it is a yes/no on a filename, and this harness has already watched
a model replace a 1,002-line module with three lines. Everything needed to show
the change was already computed: `_write` reports the line delta *afterwards*,
to the model. This says it first, to you.

Two consumers share one diff:

- the approval prompt, before the call runs (`plan`), and
- the transcript, after it ran (`diff_rows` over the real before and after), so
  under `--accept-all` a change is still shown rather than scrolling past as a
  one-line result.

Nothing here writes to disk. `plan` is a best-effort account of the work
`tools.dispatch` is about to do; where it cannot be sure (a fuzzy Edit match, a
shell command's effects) it says so rather than guessing.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

from . import tools

MAX_DIFF_ROWS = 40
MAX_NEW_FILE_LINES = 20
CONTEXT = 3


# ── What a pending call will do ─────────────────────────────────────────────
@dataclass
class Plan:
    """The pending change. `before is None` means the file does not exist yet."""
    path: str
    before: str | None = None
    after: str | None = None
    verdict: str = ""        # "", or a line that must be read before answering
    note: str = ""           # context that is not a verdict


def plan(name: str, params: dict, cfg: dict) -> Plan | None:
    """The change a Write/Edit/Bash call will make, or None for other tools."""
    if "_stopped" in params:
        return Plan(params.get("file_path", ""),
                    verdict="will be REFUSED — this call was stopped while being written")
    try:
        if name == "Write":
            return _plan_write(params, cfg)
        if name == "Edit":
            return _plan_edit(params, cfg)
        if name == "Bash":
            return _plan_bash(params, cfg)
    except Exception as e:                       # pragma: no cover - defensive
        return Plan(params.get("file_path", ""), note=f"(could not preview: {e})")
    return None


def _plan_write(params: dict, cfg: dict) -> Plan:
    path = params.get("file_path", "")
    target, why = tools._safe_target(path, cfg)
    if why:
        return Plan(path, verdict=f"will be REFUSED — {why}")
    content = params.get("content", "")
    if not target.exists():
        return Plan(path, None, content)
    before = _read(target)
    if params.get("append"):
        sep = "" if not before or before.endswith("\n") else "\n"
        return Plan(path, before, before + sep + content)
    return Plan(path, before, content)


def _plan_edit(params: dict, cfg: dict) -> Plan:
    path = params.get("file_path", "")
    target, why = tools._safe_target(path, cfg)
    if why:
        return Plan(path, verdict=f"will be REFUSED — {why}")
    if not target.exists():
        return Plan(path, verdict=f"will FAIL — {path} does not exist")

    before = _read(target)
    old, new = params.get("old_string", ""), params.get("new_string", "")
    if old not in before:
        # tools._edit falls back to a whitespace-insensitive match, and applies
        # it only when it is unique. Say which of those is about to happen
        # rather than pretending to know the result.
        _loose, where = tools._loose_find(before, old)
        if len(where) == 1 and not params.get("replace_all"):
            return Plan(path, note="old_string does not match exactly; a unique "
                        "whitespace-insensitive match will be used instead:\n"
                        + _numbered(before, where[0][0]))
        return Plan(path, verdict="will FAIL — old_string is not in the file")

    hits = before.count(old)
    if hits > 1 and not params.get("replace_all"):
        return Plan(path, verdict=f"will FAIL — old_string appears {hits} times, not once")
    after = before.replace(old, new) if params.get("replace_all") \
        else before.replace(old, new, 1)
    note = f"{hits} occurrences" if params.get("replace_all") and hits > 1 else ""
    return Plan(path, before, after, note=note)


def _plan_bash(params: dict, cfg: dict) -> Plan:
    cmd = params.get("command", "")
    why = tools.bash_outside_jail(cmd, cfg)
    if why:
        # Worth saying before you answer: approving it changes nothing, the
        # jail refuses it either way.
        return Plan("", verdict=f"will be REFUSED — {why}")
    cwd = cfg.get("_cwd") or ""
    return Plan("", note=f"runs in {cwd}" if cwd else "")


# ── The diff itself ─────────────────────────────────────────────────────────
@dataclass
class Row:
    """One rendered line. kind: "ctx" | "add" | "del" | "gap"."""
    kind: str
    old_no: int | None
    new_no: int | None
    text: str
    # (start, end) character ranges that changed within the line, for a line
    # that was edited rather than wholly added or removed.
    spans: list[tuple[int, int]] = field(default_factory=list)


def diff_rows(before: str | None, after: str, context: int = CONTEXT) -> list[Row]:
    """Line-numbered hunks, with the changed words marked inside edited lines."""
    b = (before or "").splitlines()
    a = after.splitlines()
    rows: list[Row] = []
    groups = difflib.SequenceMatcher(None, b, a, autojunk=False) \
        .get_grouped_opcodes(context)
    for gi, group in enumerate(groups):
        if gi:
            rows.append(Row("gap", None, None, "…"))
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                rows += [Row("ctx", i1 + k + 1, j1 + k + 1, b[i1 + k])
                         for k in range(i2 - i1)]
                continue
            dels = [Row("del", i1 + k + 1, None, b[i1 + k]) for k in range(i2 - i1)]
            adds = [Row("add", None, j1 + k + 1, a[j1 + k]) for k in range(j2 - j1)]
            # A replaced line paired with its replacement: mark only what
            # changed, the way an editor does. Unpaired lines stay whole.
            if tag == "replace":
                for d, n in zip(dels, adds):
                    d.spans, n.spans = _changed_spans(d.text, n.text)
            rows += dels + adds
    return rows


def _changed_spans(old: str, new: str) -> tuple[list, list]:
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    if sm.ratio() < 0.4:
        return [], []          # mostly rewritten: highlighting words is noise
    o, n = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            if i2 > i1:
                o.append((i1, i2))
            if j2 > j1:
                n.append((j1, j2))
    return o, n


def counts(before: str | None, after: str) -> tuple[int, int]:
    """(lines added, lines removed)."""
    rows = diff_rows(before, after, context=0)
    return (sum(r.kind == "add" for r in rows), sum(r.kind == "del" for r in rows))


def summary(path: str, before: str | None, after: str, pending: bool = False) -> str:
    """One line: what a change did, or — `pending` — what it is about to do."""
    if before is None:
        n = len(after.splitlines())
        return f"{'Will create' if pending else 'Created'} {path} ({n} line{'s' * (n != 1)})"
    add, rem = counts(before, after)
    if not add and not rem:
        return f"No change to {path}"
    return (f"{'Will update' if pending else 'Updated'} {path} with "
            f"{add} addition{'s' * (add != 1)} and {rem} removal{'s' * (rem != 1)}")


def gutting(before: str | None, after: str) -> str:
    """The loud line, when a write removes most of a file. Empty otherwise."""
    b, a = len((before or "").splitlines()), len(after.splitlines())
    if b >= 20 and a <= b // 2:
        return (f"!! this removes {b - a} of {b} lines "
                f"({100 * (b - a) // b}% of the file)")
    return ""


# ── Plain-text form (the prompt's fallback, and what tests read) ────────────
def describe(name: str, params: dict, cfg: dict) -> str:
    """A short human-readable account of the pending call. Never raises."""
    p = plan(name, params, cfg)
    if p is None:
        return ""
    if p.verdict:
        return p.verdict
    if p.after is None:
        return p.note
    out = []
    if p.note:
        out.append(p.note)
    if p.before is None:
        lines = p.after.splitlines()
        out.append(f"creates {p.path}, {len(lines)} lines")
        out += [f"+{l}" for l in lines[:MAX_NEW_FILE_LINES]]
        if len(lines) > MAX_NEW_FILE_LINES:
            out.append(f"… {len(lines) - MAX_NEW_FILE_LINES} more lines")
        return "\n".join(out)
    if p.before == p.after:
        return "no change"
    add, rem = counts(p.before, p.after)
    out.append(f"{len(p.before.splitlines())} lines -> "
               f"{len(p.after.splitlines())} (+{add} -{rem})")
    body = [r for r in diff_rows(p.before, p.after)]
    for r in body[:MAX_DIFF_ROWS]:
        out.append({"add": "+", "del": "-", "ctx": " ", "gap": "@@"}[r.kind] + r.text
                   if r.kind != "gap" else "@@ …")
    if len(body) > MAX_DIFF_ROWS:
        out.append(f"… {len(body) - MAX_DIFF_ROWS} more lines")
    # The case this exists for goes *last* — next to the prompt, where a long
    # diff cannot scroll it off the top of the screen.
    if (loud := gutting(p.before, p.after)):
        out.append(loud)
    return "\n".join(out)


def _numbered(text: str, start: int) -> str:
    lines = text[:start].count("\n")
    block = text[start:].splitlines()[:6]
    return "\n".join(f"{lines + i + 1:6d}\t{l}" for i, l in enumerate(block))


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
