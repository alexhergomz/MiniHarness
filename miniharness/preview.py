"""What a tool call will do, rendered for the person about to approve it.

The permission prompt used to print a tool name and a path. That is not
oversight — it is a yes/no on a filename, and this harness has already watched
a model replace a 1,002-line module with three lines. Everything needed to show
the change was already computed: `_write` reports the line delta *afterwards*,
to the model. This says it first, to you.

Nothing here touches the filesystem except to read. The preview is a
best-effort description of the same work `tools.dispatch` is about to do; where
it cannot be sure (a fuzzy Edit match, a shell command's effects) it says so
rather than guessing.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from . import tools

MAX_DIFF_LINES = 40
MAX_NEW_FILE_LINES = 20


def describe(name: str, params: dict, cfg: dict) -> str:
    """A short human-readable account of the pending call. Never raises."""
    try:
        if name == "Write":
            return _write(params, cfg)
        if name == "Edit":
            return _edit(params, cfg)
        if name == "Bash":
            return _bash(params, cfg)
    except Exception as e:                       # pragma: no cover - defensive
        return f"(could not preview: {type(e).__name__}: {e})"
    return ""


# ── Write ───────────────────────────────────────────────────────────────────
def _write(params: dict, cfg: dict) -> str:
    path = params.get("file_path", "")
    target, why = tools._safe_target(path, cfg)
    if why:
        return f"will be REFUSED — {why}"

    content = params.get("content", "")
    new_lines = content.splitlines()
    if params.get("append"):
        have = len(_read(target).splitlines()) if target.exists() else 0
        head = "\n".join(f"+{l}" for l in new_lines[:MAX_DIFF_LINES])
        more = _elided(len(new_lines), MAX_DIFF_LINES)
        return f"appends {len(new_lines)} lines to {have}\n{head}{more}"

    if not target.exists():
        head = "\n".join(f"+{l}" for l in new_lines[:MAX_NEW_FILE_LINES])
        return (f"creates {path}, {len(new_lines)} lines\n{head}"
                + _elided(len(new_lines), MAX_NEW_FILE_LINES))

    before = _read(target)
    return _diff(before, content, path)


# ── Edit ────────────────────────────────────────────────────────────────────
def _edit(params: dict, cfg: dict) -> str:
    path = params.get("file_path", "")
    target, why = tools._safe_target(path, cfg)
    if why:
        return f"will be REFUSED — {why}"
    if not target.exists():
        return f"will FAIL — {path} does not exist"

    before = _read(target)
    old = params.get("old_string", "")
    new = params.get("new_string", "")
    if old not in before:
        # tools._edit falls back to a whitespace-insensitive match, and applies
        # it only when it is unique. Say which of those is about to happen
        # rather than pretending to know the result.
        _loose, where = tools._loose_find(before, old)
        if len(where) == 1 and not params.get("replace_all"):
            return ("old_string does not match exactly; a unique "
                    "whitespace-insensitive match will be used instead:\n"
                    + _numbered(before, where[0][0]))
        return "will FAIL — old_string is not in the file"

    hits = before.count(old)
    if hits > 1 and not params.get("replace_all"):
        return f"will FAIL — old_string appears {hits} times, not once"

    after = before.replace(old, new) if params.get("replace_all") \
        else before.replace(old, new, 1)
    note = f"{hits} occurrences\n" if params.get("replace_all") and hits > 1 else ""
    return note + _diff(before, after, path)


# ── Bash ────────────────────────────────────────────────────────────────────
def _bash(params: dict, cfg: dict) -> str:
    cmd = params.get("command", "")
    why = tools.bash_outside_jail(cmd, cfg)
    if why:
        # Worth saying before you answer: approving it changes nothing, the
        # jail refuses it either way.
        return f"will be REFUSED — {why}"
    cwd = cfg.get("_cwd") or ""
    return f"runs in {cwd}" if cwd else ""


# ── Rendering ───────────────────────────────────────────────────────────────
def _diff(before: str, after: str, path: str) -> str:
    if before == after:
        return "no change"
    b, a = before.splitlines(), after.splitlines()
    body = [l.rstrip("\n") for l in difflib.unified_diff(
        b, a, lineterm="", n=2, fromfile=path, tofile=path)][2:]   # drop ---/+++

    added = sum(1 for l in body if l.startswith("+"))
    removed = sum(1 for l in body if l.startswith("-"))
    out = [f"{len(b)} lines -> {len(a)} (+{added} -{removed})",
           "\n".join(body[:MAX_DIFF_LINES]) + _elided(len(body), MAX_DIFF_LINES)]

    # The case this exists for. A write that removes most of a file is almost
    # never what was meant, so it goes *last* — next to the prompt, where a
    # long diff cannot scroll it off the top of the screen.
    if len(b) >= 20 and len(a) <= len(b) // 2:
        out.append(f"!! this removes {len(b) - len(a)} of {len(b)} lines "
                   f"({100 * (len(b) - len(a)) // len(b)}% of the file)")
    return "\n".join(out)


def _numbered(text: str, start: int) -> str:
    lines = text[:start].count("\n")
    block = text[start:].splitlines()[:6]
    return "\n".join(f"{lines + i + 1:6d}\t{l}" for i, l in enumerate(block))


def _elided(total: int, shown: int) -> str:
    return f"\n… {total - shown} more lines" if total > shown else ""


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
