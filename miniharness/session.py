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
