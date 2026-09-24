"""Session persistence: one JSONL file per session.

Promethean used SQLite with an FTS5 index. A coding session is an append-only
list of messages that gets read back in full or not at all — there is nothing to
query, so there is nothing for an index to do. JSONL is greppable, diffable,
resumable, and repairable with a text editor when something goes wrong.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import HOME

SESSIONS = HOME / "sessions"


def _slug(path: str) -> str:
    base = os.path.basename(os.path.abspath(path)) or "root"
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in base)[:40]


def new_id(cwd: str) -> str:
    return f"{_slug(cwd)}-{time.strftime('%Y%m%d-%H%M%S')}"


def path_for(session_id: str) -> Path:
    return SESSIONS / f"{session_id}.jsonl"


def append(session_id: str, record: dict) -> None:
    """Append one record. Never raises — losing a transcript line must not
    take down the agent mid-turn."""
    try:
        SESSIONS.mkdir(parents=True, exist_ok=True)
        with open(path_for(session_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def load(session_id: str) -> list[dict]:
    """Read a session's messages back. Skips corrupt lines rather than failing."""
    p = path_for(session_id)
    if not p.exists():
        return []
    out = []
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out


def recent(limit: int = 20) -> list[tuple[str, float, int]]:
    """(session_id, mtime, n_messages), newest first."""
    if not SESSIONS.exists():
        return []
    rows = []
    for p in SESSIONS.glob("*.jsonl"):
        try:
            st = p.stat()
            n = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        rows.append((p.stem, st.st_mtime, n))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:limit]


def latest() -> str | None:
    rows = recent(1)
    return rows[0][0] if rows else None


# ── Saving as it happens ────────────────────────────────────────────────────
# A session used to be written once per turn, at the end of it. A crash, a
# closed terminal or a laptop going to sleep mid-turn lost everything the model
# had done since the user's last message — in a thirty-minute build turn, all
# of it. Watched: a run cut off by a suspend kept nothing but its opening line.
#
# So the file is now a log of changes, written at every step:
#
#   {"meta": {...}}          once, when the session starts: cwd, time
#   {"add": [msg, ...]}      messages appended since the last save
#   {"messages": [...]}      a full snapshot, when history was rewritten —
#                            compaction, repair — rather than appended to
#   {"ledger": [...]}        rides along on either, for the working set
#
# Old files are all snapshots, so they replay exactly as they always loaded.
import hashlib


def _fingerprint(messages: list[dict]) -> str:
    return hashlib.sha1(json.dumps(messages, sort_keys=True, ensure_ascii=False)
                        .encode("utf-8", "replace")).hexdigest()


def save_point(state, cwd: str = "") -> None:
    """Write whatever changed in `state` since the last save. Never raises.

    Appends are written as appends. Anything else — the history was shortened
    or an earlier message changed — is written as a snapshot, so a replay can
    never reconstruct a history that did not exist.
    """
    sid = getattr(state, "session_id", "")
    if not sid:
        return
    saved_n, saved_fp, saved_ledger = getattr(state, "_saved", (0, "", 0))
    messages = state.messages
    ledger = list(getattr(state, "ledger", []) or [])
    record: dict = {}
    if not path_for(sid).exists():
        append(sid, {"meta": {"cwd": os.path.abspath(cwd or os.getcwd()),
                              "started": time.strftime("%Y-%m-%d %H:%M:%S")}})
    if (saved_n and len(messages) >= saved_n
            and _fingerprint(messages[:saved_n]) == saved_fp):
        if len(messages) > saved_n:
            record["add"] = messages[saved_n:]
    else:
        record["messages"] = messages
    if len(ledger) > saved_ledger:
        record["ledger"] = [list(e) for e in ledger[saved_ledger:]]
    if record:
        append(sid, record)
    state._saved = (len(messages), _fingerprint(messages), len(ledger))


def restore(session_id: str) -> dict:
    """Replay a session file: {"messages", "ledger", "cwd"}. Empty if none."""
    messages: list[dict] = []
    ledger: list = []
    cwd = ""
    for rec in load(session_id):
        if "meta" in rec:
            cwd = rec["meta"].get("cwd", cwd)
        if "messages" in rec:
            messages = list(rec["messages"])
        if "add" in rec:
            messages.extend(rec["add"])
        if "ledger" in rec:
            ledger.extend(tuple(e) for e in rec["ledger"])
    return {"messages": messages, "ledger": ledger, "cwd": cwd}


def unfinished(messages: list[dict]) -> bool:
    """True if the last turn stopped before the model answered.

    A finished turn ends with the model's own reply and no tool calls pending.
    Ending on a tool result, on the user's message, or on tool calls with no
    results means it was cut off mid-way.
    """
    if not messages:
        return False
    last = messages[-1]
    return last.get("role") != "assistant" or bool(last.get("tool_calls"))


def latest_for(cwd: str) -> str | None:
    """The most recent session started in this directory."""
    want = os.path.abspath(cwd)
    for sid, _mtime, _n in recent(200):
        first = next(iter(load(sid)), {})
        where = first.get("meta", {}).get("cwd") if "meta" in first else None
        if where == want or (where is None and sid.startswith(_slug(cwd) + "-")):
            return sid
    return None
