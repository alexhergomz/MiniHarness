"""A checkpoint after every accepted change, in a repository of our own.

The agent wrecks a file and there is no way back. `/undo` covers the last 50
Write/Edit calls in memory, but nothing covers a Bash command that deleted a
directory, and nothing survives the process exiting. Watched: after destroying
a 1,002-line module the model ran `git checkout` — against the user's real
repository, which is the wrong repository *and* would have taken their own
uncommitted work with it.

So the checkpoints live in a **shadow repository**: a git dir under
`~/.miniharness/checkpoints`, pointed at the working directory with
`--work-tree`. The user's `.git` is never opened, their index is never staged,
their branch never moves, their history gains nothing. A project that is not a
git repository at all gets checkpoints just the same — that is the case with
the most to lose.

Nothing here raises. A failed checkpoint is worth a notice, never a dead turn.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from .config import HOME

STORE = HOME / "checkpoints"

# Written to the shadow repo's info/exclude. The working tree may not be a git
# repository, so there may be no .gitignore to inherit, and a checkpoint that
# stages .venv or node_modules takes long enough that people turn the feature
# off. `.git` itself git already refuses to add.
EXCLUDES = """\
.miniharness/
.venv/
venv/
node_modules/
__pycache__/
*.pyc
.mypy_cache/
.pytest_cache/
.ruff_cache/
target/
dist/
build/
*.egg-info/
"""

SLOW_SECONDS = 2.0          # past this, say so once rather than silently stall
_IDENTITY = ("-c", "user.name=miniharness", "-c", "user.email=miniharness@local",
             "-c", "commit.gpgsign=false")


def available() -> bool:
    return shutil.which("git") is not None


def enabled(cfg: dict) -> bool:
    return bool(cfg.get("checkpoints", True)) and available()


def _cwd(cfg: dict) -> str:
    return cfg.get("_cwd") or os.getcwd()


def store_for(cfg: dict, session_id: str) -> Path:
    slug = "".join(c if c.isalnum() or c in "-_" else "-"
                   for c in os.path.basename(os.path.abspath(_cwd(cfg))))[:40]
    return STORE / f"{slug or 'root'}-{session_id or 'adhoc'}.git"


def _git(cfg: dict, session_id: str, *args: str,
         timeout: float = 60.0) -> subprocess.CompletedProcess | None:
    """Run one git command against the shadow repo. None if it could not run."""
    gitdir = store_for(cfg, session_id)
    try:
        return subprocess.run(
            ["git", f"--git-dir={gitdir}", f"--work-tree={_cwd(cfg)}", *args],
            capture_output=True, text=True, timeout=timeout,
            # Somebody else's GIT_DIR in the environment would silently redirect
            # every one of these at their repository.
            env={k: v for k, v in os.environ.items()
                 if not k.startswith(("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX"))})
    except (OSError, subprocess.SubprocessError):
        return None


def _init(cfg: dict, session_id: str) -> bool:
    gitdir = store_for(cfg, session_id)
    if (gitdir / "HEAD").exists():
        return True
    try:
        gitdir.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "init", "-q", "--bare", str(gitdir)],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False
        (gitdir / "info").mkdir(exist_ok=True)
        excludes = EXCLUDES + _exclude_self(gitdir, _cwd(cfg))
        (gitdir / "info" / "exclude").write_text(excludes, encoding="utf-8")
    except (OSError, subprocess.SubprocessError):
        return False
    # A bare repo refuses work-tree operations until told it is not bare.
    return _git(cfg, session_id, "config", "core.bare", "false") is not None


def _exclude_self(gitdir: Path, work: str) -> str:
    """Keep the store out of any tree that happens to contain it.

    ~/.miniharness sits inside the home directory, so an agent pointed at $HOME
    would snapshot its own checkpoints into itself — and a rewind would then
    delete the checkpoints it was rewinding to. Found by a test that put the
    store inside the working tree; the same thing is one `cd ~` away in real
    use.
    """
    try:
        rel = os.path.relpath(gitdir.parent, work)
    except ValueError:                          # different drives, on Windows
        return ""
    return "" if rel.startswith("..") else f"/{rel.replace(os.sep, '/')}/\n"


def snapshot(cfg: dict, session_id: str, label: str) -> tuple[str, str] | None:
    """Commit the working tree. Returns (short_sha, note) or None if unchanged.

    `note` is empty unless there is something worth telling the user, which is
    currently only that the snapshot was slow enough to notice.
    """
    if not enabled(cfg) or not _init(cfg, session_id):
        return None
    t0 = time.monotonic()
    if _git(cfg, session_id, "add", "-A") is None:
        return None
    r = _git(cfg, session_id, *_IDENTITY, "commit", "-q",
             "--allow-empty-message", "-m", label[:200])
    if r is None or r.returncode != 0:
        return None                     # nothing staged: nothing changed
    head = _git(cfg, session_id, "rev-parse", "--short", "HEAD")
    if head is None or head.returncode != 0:
        return None
    spent = time.monotonic() - t0
    note = (f"checkpointing this directory takes {spent:.0f}s — "
            f"/config checkpoints=false turns it off" if spent > SLOW_SECONDS else "")
    return head.stdout.strip(), note


def ensure_baseline(cfg: dict, session_id: str) -> None:
    """Checkpoint the untouched tree before the first change is made.

    Called on the first mutating call rather than at startup, so a session that
    only reads never pays for it — but called *before* that call runs, because
    "the state before the agent touched anything" is the checkpoint people
    actually want and it cannot be reconstructed afterwards.
    """
    if not enabled(cfg):
        return
    if _init(cfg, session_id) and not history(cfg, session_id, limit=1):
        snapshot(cfg, session_id, "before the first change")


def history(cfg: dict, session_id: str, limit: int = 20) -> list[tuple[str, str, str]]:
    """[(short_sha, relative_time, label)], newest first."""
    if not available():
        return []
    r = _git(cfg, session_id, "log", f"-{limit}",
             "--format=%h\t%cr\t%s", timeout=20)
    if r is None or r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            out.append((parts[0], parts[1], parts[2]))
    return out


def changes_since(cfg: dict, session_id: str, ref: str) -> str:
    """What restoring `ref` would undo, as a stat. Empty when nothing would."""
    if not available():
        return ""
    r = _git(cfg, session_id, "diff", "--stat", ref, timeout=30)
    # strip("\n") rather than strip(): git indents every line of a --stat, and
    # dropping the first line's indent alone breaks the column alignment.
    return r.stdout.strip("\n") if r and r.returncode == 0 else ""


def restore(cfg: dict, session_id: str, ref: str) -> str:
    """Put the working tree back to `ref`. Returns a human-readable result.

    This deletes files created since the checkpoint — that is the point, and it
    is why the caller shows `changes_since` and asks first.
    """
    if not enabled(cfg):
        return "Checkpoints are off."
    r = _git(cfg, session_id, "read-tree", "-u", "--reset", ref, timeout=120)
    if r is None:
        return "Could not run git."
    if r.returncode != 0:
        return f"Could not restore {ref}: {r.stderr.strip()[:200]}"
    return f"Working tree restored to {ref}."
