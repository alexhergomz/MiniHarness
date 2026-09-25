"""The eight tools.

Promethean registered 34. Every tool pays its schema token cost on *every turn
of every session forever*, so the bar for inclusion is high: a tool ships only
if the model measurably cannot do the job with the tools already present.

    Read Write Edit Bash Glob Grep WebFetch WebSearch

Notably absent, and why (DESIGN.md §1): Think / TaskCreate / EnterPlanMode
(reasoning ceremony a 9B model can't afford), GetCallers / Outline /
Neighborhood / PathBetween / Imports / SearchFiles (six tools folded into
FindSymbol), GetDiagnostics / NotebookEdit (Bash covers them), and the whole
email / spreadsheet / browser surface (not a coding agent's job).

The safety layer is honest about what it is: a guardrail against a confused
model, not a sandbox. Run untrusted models in a container.
"""

from __future__ import annotations

import fnmatch
import json
from collections import OrderedDict
import os
import re
import shutil
import subprocess
import sys
import time as _time
from pathlib import Path
from typing import Any

MAX_OUTPUT = 30_000  # chars; beyond this a tool result is costing more than it returns

# No single observation may occupy more than this share of the context window.
# A fixed 30,000-character cap is ~7,500 tokens, which is larger than an entire
# 4k window: one Read could then make the request unrecoverable, and compaction
# could not help because the oversized result was the newest thing in the
# conversation. Capping relative to the window fixes it at the source.
def output_cap(config: dict) -> int:
    """Character budget for one tool result, as a share of the window."""
    from . import config as _cfg
    return min(MAX_OUTPUT, _cfg.budget(config, "tool_output_share", floor=500) * 4)


# ── Safety ──────────────────────────────────────────────────────────────────
# Fires even under accept_all. These are the commands with no plausible benign
# reading in an agent loop.
_DANGEROUS = [
    # Recursive rm whose *target* is a root-ish path. Deliberately not
    # "any recursive rm": `rm -rf build/` is an ordinary thing to run, and a
    # deny-list that cries wolf gets disabled.
    r"\brm\s+(?:-[\w-]+\s+)*-[\w]*[rf][\w]*\s+(?:-[\w-]+\s+)*(?:/|/\*|~|~/\*?|\$HOME)(?:\s|$)",
    r"\brm\s+(?:-[\w-]+\s+)*-[\w]*[rf][\w]*\s+(?:-[\w-]+\s+)*"
    r"/(?:bin|boot|dev|etc|home|lib|lib64|opt|proc|root|sbin|srv|sys|usr|var)\b",
    r"\bdd\b.*\bof=/dev/(sd|nvme|hd|disk)",
    r"\bmkfs(\.\w+)?\b",
    r">\s*/dev/(sd|nvme|hd|disk)",
    r":\(\)\s*\{.*\|.*&.*\}\s*;?\s*:",          # fork bomb
    r"\bchmod\s+(-[a-zA-Z]+\s+)*777\s+/\s*$",
    r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh",
    r"\bshutdown\b|\breboot\b|\bhalt\b",
    r"\bgit\s+push\b.*--force.*\b(main|master)\b",
]
_DANGEROUS_RE = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS]

# Blocked unconditionally for Read/Write/Edit, in every mode.
# Belt and braces inside the jail: a repository can contain its own secrets,
# and the working directory may legitimately be a home directory.
_SENSITIVE = [
    "~/.ssh", "~/.aws", "~/.gnupg", "~/.config/gcloud", "~/.kube",
    "/etc/shadow", "/etc/sudoers", "~/.netrc", "~/.npmrc", "~/.pypirc",
    "~/.docker/config.json", "~/.git-credentials",
]


def is_dangerous_bash(cmd: str) -> str | None:
    """Return a reason string if the command is refused outright, else None."""
    for rx in _DANGEROUS_RE:
        if rx.search(cmd):
            return f"blocked by deny-list (matched {rx.pattern!r})"
    return None


# Directories a command may touch even though they are outside the jail:
# the toolchain itself. Read-only in practice — nothing the agent is asked to
# do involves writing here, and a write would fail on permissions anyway.
# Deliberately narrow: the toolchain, and nothing that holds configuration or
# credentials. /etc is NOT here — allowlisting it let `../../etc/shadow`
# through in testing. Tools that read /etc do so by themselves; the agent has
# no reason to name it.
_SYSTEM_ROOTS = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/opt",
                 "/snap")

# Anything that looks like a path: absolute, ~-relative, or explicitly
# ./ or ../ prefixed. Bare words are not treated as paths — `pytest` is a
# command, not a file — but anything with a separator is checked.
_PATHLIKE_RE = re.compile(r"""(?<![\w:/])(~|\.{1,2})?/[^\s;|&<>()'"`]*""")
# Stripped before scanning: the path part of a URL is not a filesystem path,
# and matching it refused every `curl https://...` in testing.
_URL_RE = re.compile(r"""\b\w+://[^\s;|&<>()'"`]*""")


def bash_outside_jail(cmd: str, cfg: dict) -> str | None:
    """Refuse a command that names a path outside the working directory.

    The jail used to cover Read/Write/Edit/Glob/Grep and stop at Bash, on the
    reasoning that confining a shell means parsing arbitrary commands. That
    left the jail decorative: everything it protected was reachable with `cat`.

    It stopped being theoretical. An agent under test wrecked a file, had no
    way to undo it, and ran

        cd /home/<user>/Desktop/MiniHarness && git checkout HEAD -- miniharness

    against the repository the sandbox had been copied from. It failed only
    because the model typed the path with a space in it. Had it not, it would
    have reverted uncommitted work outside its sandbox.

    So paths are checked, and the check is deliberately blunt: any token with a
    separator in it must resolve inside a jail root or the toolchain. The only
    exceptions are the ones the user configures (`extra_roots`).

    **This is not a security boundary and must not be described as one.** A
    shell can build a path the parser never sees — `$(printf ...)`, a variable,
    an `xargs`. It raises the cost of leaving the jail from zero to deliberate.
    Real confinement needs a container, or `bwrap` with unprivileged user
    namespaces available (on Ubuntu, `kernel.apparmor_restrict_unprivileged_userns`
    blocks it by default).
    """
    roots = jail_roots(cfg)
    home = Path.home()
    for m in _PATHLIKE_RE.finditer(_URL_RE.sub(" ", cmd)):
        token = m.group(0).rstrip(".,;:")
        if "://" in token or token.startswith("//"):
            continue                       # a URL, not a path
        # A lone slash is arithmetic, not a path. Numerical code is full of
        # `a / b`, and matching it refused a `python3 -c` mid-run on the ML
        # build — for a data-science harness that is fatal. `ls /` is given up
        # with it; the syscall watcher still sees any actual access.
        if not token.strip("/") or not any(c.isalnum() for c in token):
            continue
        try:
            target = Path(token).expanduser()
            if not target.is_absolute():
                target = Path(cfg.get("_cwd") or os.getcwd()) / target
            target = target.resolve()
        except (OSError, RuntimeError):
            continue
        if any(target == r or r in target.parents for r in roots):
            continue
        if any(str(target).startswith(s + "/") or str(target) == s
               for s in _SYSTEM_ROOTS):
            continue
        # Same allowances the syscall watcher makes. They were separate lists,
        # and the static check refused `2>/dev/null` — one of the commonest
        # idioms in a shell — because only the watcher knew about /dev.
        #
        # The sensitive list is consulted first, exactly as the watcher does.
        # Without it, widening to /etc let /etc/shadow through — caught by the
        # tests within a minute, which is the argument for having written them.
        t = str(target)
        if is_sensitive_path(t):
            return (f"{token} is a sensitive path and is never readable, "
                    f"inside the working directory or out.")
        if any(t.startswith(i) or t == i.rstrip("/") for i in _INCIDENTAL):
            continue
        # Flat files directly under a scratch root only — the same rule the
        # watcher uses. A permissive prefix here opened every nested path under
        # /tmp, which the jail tests caught immediately.
        if any(t == r or (t.startswith(r + "/") and "/" not in t[len(r) + 1:])
               for r in _scratch_roots()):
            continue
        # Everything else is refused, and the user's own files most of all.
        where = "your home directory" if (target == home or home in target.parents) \
            else "outside the working directory"
        return (f"{token} is {where}. Commands may only touch "
                f"{', '.join(str(r) for r in roots)} — add another with "
                f"`extra_roots` if that is wrong.")
    return None


def _toolchain_roots() -> tuple[str, ...]:
    """Where the interpreter and its libraries live.

    Not a concession — a necessity. Measured on one `pytest` run inside a jail:
    1,225 distinct paths outside it, of which 1,205 were the Python
    installation, because on this machine Python lives under the user's home.
    A jail that forbids the toolchain forbids running anything.
    """
    import sysconfig
    roots = {sys.base_prefix, sys.prefix, sysconfig.get_paths()["purelib"]}
    roots |= {p for p in sys.path if p.startswith("/")}
    return tuple(sorted(r.rstrip("/") for r in roots if r))


# Read-only noise every process makes: the loader, the terminal, the clock.
# Measured from the same run — with the toolchain excluded, 25 foreign paths
# remained and all of them were of this kind.
# /etc entire, read-only. Enumerating its files one at a time was wrong three
# times over: ld.so and localtime, then gitconfig, then openssl.cnf the moment
# the scientific stack was imported. Every tool reads its configuration from
# somewhere under /etc and the list has no end.
#
# Safe because `is_sensitive_path` is consulted first and holds the things that
# actually matter there — /etc/shadow, /etc/sudoers — and because nothing under
# /etc is ever writable through this path.
_INCIDENTAL = ("/dev/", "/proc/", "/sys/", "/etc/", "/run/", "/var/lib/dbus/")

# A tool's own configuration, read-only. Derived rather than guessed: 14
# representative commands were run under the watcher and every denial
# collected. After the toolchain and scratch space, exactly these remained —
# git reading its config, and /dev/null. Three earlier false positives were
# each patched from a single symptom; this list came from the whole workload.
_TOOL_CONFIG_HOME = (".gitconfig", ".config/git")


def _scratch_roots() -> tuple[str, ...]:
    """The system temp directory, read and write.

    A shell makes its own temp files — heredocs, process substitution, pipes —
    and killing the command for touching them refused `pytest` a second time,
    one fix after __pycache__. Temp is scratch space for every process on the
    machine and is treated as such here.

    The trade is explicit: a command can read whatever else is in /tmp. The
    jail protects the user's files, not the machine's scratch area, and
    anything secret should not be in /tmp to begin with. `_SENSITIVE` still
    applies wherever the path lives.
    """
    import tempfile
    # /dev/shm is scratch too: numpy and scikit-learn create POSIX semaphores
    # there for multiprocessing, which is a *write*, and denying it refused
    # `import pandas, sklearn` outright.
    return tuple({tempfile.gettempdir().rstrip("/"), "/tmp", "/var/tmp",
                  "/dev/shm"})
_INCIDENTAL_HOME = (".inputrc", ".terminfo", ".profile") + _TOOL_CONFIG_HOME

_STRACE_PATH_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
# Only calls that open, execute or modify. stat/access/readlink are skipped:
# a shell stats its own parents constantly, and treating that as an escape
# refused `echo hi && cat hello.txt` on the first try.
_WATCHED_SYSCALLS = frozenset((
    "open", "openat", "openat2", "creat", "execve", "execveat",
    "unlink", "unlinkat", "rename", "renameat", "renameat2",
    "mkdir", "mkdirat", "rmdir", "truncate", "chmod", "fchmodat",
    "chown", "fchownat", "link", "linkat", "symlink", "symlinkat"))
_WRITE_SYSCALLS = ("open", "creat", "unlink", "rename", "mkdir", "rmdir",
                   "truncate", "chmod", "chown", "link", "symlink")


def _access_allowed(path: str, syscall: str, line: str, roots, tool) -> bool:
    """Is this one filesystem access acceptable from inside the jail?"""
    if not path.startswith("/"):
        return True                      # relative: resolved against a cwd we set
    if any(path == str(r) or path.startswith(str(r).rstrip("/") + "/") for r in roots):
        return True
    if is_sensitive_path(path):
        return False                     # never, wherever it lives
    # Flat files directly under the temp root only — /tmp/shXXXX, not
    # /tmp/someone/else/data. The shell's own scratch is always flat; anything
    # nested is somebody's directory and stays behind the jail. Allowlisting
    # the whole of /tmp instead made the jail stop protecting test fixtures,
    # which is how narrow this had to become.
    for t in _scratch_roots():
        # The root itself is listable — opening /tmp to read its entries is
        # what `ls`, mkstemp and half the toolchain do first.
        if path == t:
            return True
        if path.startswith(t + "/") and "/" not in path[len(t) + 1:]:
            return True
    if any(path.startswith(t) for t in _INCIDENTAL) or \
            any(path.startswith(t) for t in tool):
        # Toolchain and device noise: reads are fine, writes are not — except
        # bytecode. Importing anything writes __pycache__ next to the module,
        # so denying it refused every `pytest` the agent ran. Caught in the
        # first run after the watcher shipped, which is the failure mode this
        # guard has to avoid: one that cries wolf gets switched off.
        if "__pycache__" in path or path.endswith((".pyc", ".pyo")):
            return True
        if path == "/dev/null":
            return True                  # every shell writes here
        wants_write = syscall.startswith(_WRITE_SYSCALLS) and (
            "O_WRONLY" in line or "O_RDWR" in line or "O_CREAT" in line
            or not syscall.startswith("open"))
        return not wants_write
    home = str(Path.home())
    if path.startswith(home + "/"):
        rest = path[len(home) + 1:]
        return any(rest == n or rest.startswith(n + "/") for n in _INCIDENTAL_HOME)
    return any(path.startswith(s + "/") or path == s for s in _SYSTEM_ROOTS)


def watch_command(cmd: str, cfg: dict, timeout: int):
    """Run a command under strace and kill it the moment it leaves the jail.

    The static check in `bash_outside_jail` reads the command text, so a path
    the shell builds at runtime — `$(printf ...)`, a variable, an `xargs` — goes
    straight past it. This watches the syscalls instead, which is what actually
    touches the disk.

    Not a substitute for a container: an access is observed as it happens, so a
    read of one file can complete before the process is killed. What it
    guarantees for development is that the command does not *continue* outside
    its working directory, and that the attempt is reported rather than silent.

    Returns (stdout, stderr, returncode, violation) or None when strace is
    unavailable, in which case the caller falls back to the static check alone.
    """
    if not shutil.which("strace"):
        return None
    roots, tool = jail_roots(cfg), _toolchain_roots()
    cwd = cfg.get("_cwd") or os.getcwd()

    import tempfile, threading
    log = tempfile.NamedTemporaryFile(prefix="mh-watch-", suffix=".log", delete=False)
    log.close()
    try:
        proc = subprocess.Popen(
            ["strace", "-f", "-qq", "-e", "trace=%file", "-o", log.name,
             "/bin/sh", "-c", cmd],
            cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
    except OSError:
        os.unlink(log.name)
        return None

    found: list[str] = []

    def watch():
        seen = 0
        while True:
            # Read the log *after* noting whether strace has exited, never
            # before. strace buffers into the file, so the lines describing the
            # last syscalls land only as it exits; a watcher that checked
            # `poll()` after its read returned on a short command before those
            # lines existed, and the escape went unreported. Sampling `done`
            # first means the final pass always covers a complete log.
            done = proc.poll() is not None
            try:
                with open(log.name, errors="replace") as fh:
                    fh.seek(seen)
                    chunk = fh.read()
                    seen = fh.tell()
            except OSError:
                return
            for line in chunk.splitlines():
                # A syscall that failed touched nothing. This is mostly $PATH
                # search: the shell execve()s every directory in turn until one
                # works, so `python3` alone produced an "escape" to
                # ~/<a $PATH entry>/bin/python3 that does not exist. Errors are also
                # how a program asks whether a file is there at all.
                if "= -1 " in line:
                    continue
                call = line.split("(", 1)[0].split()[-1] if "(" in line else ""
                if call not in _WATCHED_SYSCALLS:
                    continue
                m = _STRACE_PATH_RE.search(line)
                if not m:
                    continue
                path = m.group(1)
                if not path.startswith("/"):
                    path = os.path.normpath(os.path.join(cwd, path))
                if not _access_allowed(path, call, line, roots, tool):
                    found.append(f"{call} {path}")
                    try:
                        os.killpg(os.getpgid(proc.pid), 9)
                    except OSError:
                        pass
                    return
            if done:
                return
            _time.sleep(0.05)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except OSError:
            pass
        out, err = proc.communicate()
        t.join(timeout=1)
        os.unlink(log.name)
        raise
    # The watcher now always makes one final pass over the completed log, so
    # this join is waiting on a bounded scan, not on a poll loop. Give it room:
    # returning before it finishes is the same silent miss as not watching.
    t.join(timeout=5)
    try:
        os.unlink(log.name)
    except OSError:
        pass
    return out, err, proc.returncode, (found[0] if found else None)


def is_sensitive_path(path: str) -> str | None:
    """Return a reason string if the path is off-limits, else None."""
    try:
        target = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    for pat in _SENSITIVE:
        base = Path(pat).expanduser()
        try:
            base = base.resolve()
        except (OSError, RuntimeError):
            continue
        if target == base or base in target.parents:
            return f"sensitive path ({pat})"
    return None


def jail_roots(cfg: dict) -> list[Path]:
    """Directories the file tools may touch.

    The working directory, plus anything explicitly opened up in
    ``extra_roots``. A build that genuinely needs ``/usr/include`` reaches it
    through Bash and its compiler, not through Read, so the jail costs nothing
    in normal use.
    """
    roots = [Path(cfg.get("_cwd") or os.getcwd())]
    for extra in cfg.get("extra_roots") or []:
        # Only real, non-empty paths. A malformed entry used to become a root:
        # extra_roots=[None] produced "<cwd>/None", str(None) resolved against
        # the working directory. Silently widening a jail from bad input is the
        # wrong direction to fail in.
        if not isinstance(extra, (str, Path)) or not str(extra).strip():
            continue
        roots.append(Path(str(extra)).expanduser())
    out = []
    for r in roots:
        try:
            resolved = r.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved not in out:
            out.append(resolved)
    return out


def _outside_jail(target: Path, cfg: dict) -> str | None:
    roots = jail_roots(cfg)
    for root in roots:
        if target == root or root in target.parents:
            return None
    shown = ", ".join(str(r) for r in roots)
    return f"outside the working directory ({shown})"


def _safe_target(path: str, cfg: dict) -> tuple[Path, str | None]:
    """Resolve exactly as the tool will, then judge *that*.

    The check and the access used to resolve differently: ``is_sensitive_path``
    took the raw string and resolved it against the *process* cwd, while the
    read opened ``_resolve(path, cfg)`` against the *agent's* cwd. So a direct
    read of ``~/.ssh/id_rsa`` was refused while ``sshlink/id_rsa`` — a symlink
    inside the working directory — resolved to a different path for the check
    than for the open, and leaked the file. Same for ``../../.ssh/id_rsa``.

    Resolving follows symlinks, so judging the resolved path closes both routes.
    """
    target = _resolve(path, cfg)
    try:
        target = target.resolve()
    except (OSError, RuntimeError):
        pass
    return target, (_outside_jail(target, cfg) or is_sensitive_path(str(target)))


def _resolve(path: str, cfg: dict) -> Path:
    """Resolve a tool path against the *agent's* working directory.

    Models overwhelmingly write relative paths (`calc.py`, `src/main.rs`).
    Resolving those against the harness process's cwd instead of the directory
    the agent was pointed at makes every relative Read fail, and the model
    burns turns re-deriving absolute paths. Found by running a real 4B model.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return Path(cfg.get("_cwd") or os.getcwd()) / p


def _truncate(s: str, limit: int = MAX_OUTPUT) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n\n[... truncated, {len(s) - limit} more chars]"


# ── Undo ────────────────────────────────────────────────────────────────────
# A stack of (path, prior_content_or_None). Every Write/Edit pushes one entry
# before touching the file; /undo pops. In-memory and session-scoped on purpose:
# git is the real undo for a coding agent, and this only has to cover the case
# where the agent clobbered something before you had a chance to commit.
_UNDO: list[tuple[str, str | None]] = []
UNDO_LIMIT = 50


def _push_undo(path: Path) -> None:
    prior = None
    if path.exists():
        try:
            prior = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
    _UNDO.append((str(path), prior))
    del _UNDO[:-UNDO_LIMIT]


def undo_last() -> str:
    """Revert the most recent Write/Edit. Returns a human-readable result."""
    if not _UNDO:
        return "Nothing to undo."
    path, prior = _UNDO.pop()
    p = Path(path)
    try:
        if prior is None:
            if p.exists():
                p.unlink()
            return f"Removed {path} (it did not exist before)"
        p.write_text(prior, encoding="utf-8")
        return f"Reverted {path}"
    except OSError as e:
        return f"Could not undo {path}: {e}"


# ── Read / Write / Edit ─────────────────────────────────────────────────────
_DEF_RE = re.compile(
    r"^\s*(?:async\s+)?(?:def|class|function|fn|func|impl|struct|enum|trait|"
    r"interface|type|const|export\s+(?:default\s+)?(?:function|class|const))\b"
    r"|^[A-Za-z_][\w]*\s*=\s*(?:lambda|function)\b"
    r"|^\s*(?:public|private|protected)\s+[\w<>\[\]]+\s+\w+\s*\(")


def _outline(lines: list[str], limit: int = 60) -> list[str]:
    """Definition lines with their numbers — what a person skims first."""
    out = []
    for i, ln in enumerate(lines, 1):
        if _DEF_RE.search(ln):
            out.append(f"{i:6d}\t{ln.rstrip()[:100]}")
            if len(out) >= limit:
                out.append(f"       … more definitions below")
                break
    return out


def _read(p: dict, cfg: dict, tracker) -> str:
    path = p["file_path"]
    f, why = _safe_target(path, cfg)
    if why:
        return f"Error: refused to read {path} — {why}"
    if not f.exists():
        return f"Error: {path} does not exist"
    if f.is_dir():
        return f"Error: {path} is a directory (use Glob or Bash ls)"
    try:
        text = f.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading {path}: {e}"

    lines = text.splitlines()
    offset = max(0, int(p.get("offset", 0) or 0))
    limit = int(p.get("limit", 2000) or 2000)
    window = lines[offset:offset + limit]
    if tracker is not None:
        tracker.mark_read(str(f))

    if not window:
        return "[empty file]"

    # A bare read of a large file is almost never what is wanted.
    #
    # The model asks for a whole file, the result is several thousand tokens, and
    # the next compaction throws it away — so it asks again. Watched live on an
    # 870-line module: two bare reads of the same file, each triggering a
    # compaction that reclaimed >20k tokens.
    #
    # So answer a bare read of a big file with the file's *shape* — its
    # definitions with line numbers — plus the opening lines. That is what a
    # person skims to decide where to look, and it makes the specific read
    # obvious. An explicit offset or limit means the model has already decided,
    # and is honoured exactly.
    asked_for_range = p.get("offset") is not None or p.get("limit") is not None
    cap = output_cap(cfg)
    whole = sum(len(ln) + 8 for ln in window)
    if not asked_for_range and whole > cap:
        head = window[:40]
        body = "\n".join(f"{i + 1:6d}\t{ln}" for i, ln in enumerate(head))
        outline = _outline(lines)
        note = [f"[{path}: {len(lines)} lines, ~{whole // 4} tokens — too large to "
                f"show in full.",
                f" Above: the first {len(head)} lines."]
        if outline:
            note.append(" Below: every definition and its line number. Read with "
                        "`offset` to jump to one, or Grep with `path` set to this "
                        "file to search inside it.")
        else:
            note.append(" Read with `offset` and `limit` to see a section, or Grep "
                        "with `path` set to this file.")
        note.append(" To read it all anyway, pass limit explicitly.]")
        # Guidance first, then structure, then the opening lines.
        #
        # This summary is itself subject to the output cap, and at a small
        # tool_output_share the cap can fall inside it. With the note last, the
        # pagination cut it off and handed back a bare prefix with no
        # explanation — the exact failure this path exists to prevent. Whatever
        # survives truncation must be the part that tells the model what to do.
        parts = ["\n".join(note)]
        if outline:
            parts.append("definitions:\n" + "\n".join(outline))
        parts.append(body)
        return "\n\n".join(parts)

    body = "\n".join(f"{i + offset + 1:6d}\t{ln}" for i, ln in enumerate(window))
    remaining = len(lines) - (offset + len(window))
    if remaining > 0:
        body += (f"\n\n[{remaining} more lines beyond the requested limit; "
                 f"use Read offset={offset + len(window)}]")
    return body


def _write(p: dict, cfg: dict, tracker) -> str:
    """Write a file, or append to one.

    Appending exists because adding to the end of a large file had no
    affordance at all. Watched live: asked to add a test to a 1,700-line
    tests/test_core.py, the model spent 23 of its 45 turns grepping for a
    unique anchor to hang an Edit on, and ran out of turns having wired both
    the config and the tool correctly. Every other part of the task was done.

    So append is its own mode, and deliberately does not require the file to
    have been read: that requirement exists to stop a model destroying content
    it never saw, and an append destroys nothing.
    """
    path, content = p["file_path"], p.get("content", "")
    append = bool(p.get("append"))
    f, why = _safe_target(path, cfg)
    if why:
        return f"Error: refused to write {path} — {why}"
    # Read-before-overwrite: the single most effective guard against a model
    # blowing away a file it never looked at. It does not apply to an append,
    # which only adds.
    if (f.exists() and not append and tracker is not None
            and not tracker.has_read(str(f))):
        return (f"Error: {path} exists but has not been read in this session. "
                f"Read it first, use `append` to add to the end, or use Edit "
                f"for a targeted change.")
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        existed = f.exists()
        before = 0
        prior = None
        if existed:
            try:
                prior = f.read_text(encoding="utf-8", errors="replace")
                before = len(prior.splitlines())
            except OSError:
                before = 0
        _push_undo(f)
        if append and existed:
            old = f.read_text(encoding="utf-8", errors="replace")
            sep = "" if old.endswith("\n") or not old else "\n"
            f.write_text(old + sep + content, encoding="utf-8")
        else:
            f.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"Error writing {path}: {e}"
    if tracker is not None and not append:
        tracker.mark_read(str(f))
    n = len(content.splitlines())
    note = _rewrite_streak(str(f), cfg)
    if not append and prior is not None and prior == content:
        # Watched on a live run: 22 of 28 calls rewrote a file with exactly
        # the content it already had, and every one was told "Updated". The
        # harness knew the bytes were identical and said something else, so
        # to the model each write looked like it had done something.
        return (f"No change: {path} already contains exactly this ({n} lines), "
                f"so writing it again did nothing. To make progress, change "
                f"something, or run the tests to see where things stand." + note)
    if append and existed:
        return f"Appended {n} lines to {path}"
    if not existed:
        return f"Created {path} ({n} lines)" + note
    # Say what was replaced when a write shrinks a file dramatically.
    #
    # Observed: a Write put a 3-line fragment over a 1,002-line module. The
    # read-before-overwrite guard did not fire — the model *had* read the file
    # — and the result said only "Updated tools.py (3 lines)", so nothing in
    # the transcript indicated the module was gone. The suite then failed to
    # collect, the model could not tell why, and it spent its remaining turns
    # trying to `git checkout` a repository outside its working directory to
    # get the file back.
    #
    # This is a statement of what happened, not a refusal. The write is the
    # model's call and sometimes replacing a file wholesale is right; being
    # told it just removed 999 lines is what lets it notice when it wasn't.
    if before and n < before // 2:
        return (f"Updated {path} ({n} lines, replacing {before}). "
                f"This removed {before - n} lines — if that was not intended, "
                f"rewrite the file with its full contents." + note)
    return f"Updated {path} ({n} lines)" + note


# Consecutive whole-file writes per path, reset by any other tool call.
_WRITE_STREAK: dict[str, int] = {}


def _rewrite_streak(path: str, cfg: dict) -> str:
    """Say when the same file is being rewritten over and over.

    Watched live on a build task: eleven consecutive Writes of the same module,
    forty seconds apart, with two test runs in the whole session. The model was
    revising code it had never run, so nothing it changed could be confirmed or
    refuted, and the acceptance score went *down* while it worked.

    Not duplicate suppression — the write happens, every time, and the count is
    reported rather than the call refused. What is missing at that point is
    evidence, so the note asks for the one thing that would supply it.
    """
    _WRITE_STREAK[path] = _WRITE_STREAK.get(path, 0) + 1
    for other in list(_WRITE_STREAK):
        if other != path:
            _WRITE_STREAK.pop(other, None)
    streak = _WRITE_STREAK[path]
    if streak < 3:
        return ""
    return (f"\n[this is {streak} writes to this file in a row with nothing run "
            f"in between — run the tests or the code, so the next change is "
            f"based on what actually happened]")


def note_other_tool_use(name: str) -> None:
    """Any non-Write call ends the write streak: the model went and looked at
    something. Anything that changes or runs something ends the read streak."""
    if name != "Write":
        _WRITE_STREAK.clear()
    if name in MUTATING:
        _READ_STREAK.clear()
    if name in ("Write", "Edit"):
        _FAIL_STREAK["edited"] = True


# path -> (reads since anything was changed or run, file content hash)
_READ_STREAK: dict[str, tuple[int, int]] = {}


def _reread_streak(path: str) -> str:
    """Say when a file is being re-read with nothing done in between.

    The mirror of _rewrite_streak. Watched live: six Reads of the same module
    in a row — 31, 31, 21, 21 lines — with no edit and no test run between
    them. The model was looking again at text it already had, unchanged, and
    the harness knew both of those things and said neither.

    Not duplicate suppression: every read is served in full. The note only
    says what is true — the file has not changed — and names the step that
    would produce something new.
    """
    try:
        digest = hash(Path(path).read_bytes())
    except OSError:
        return ""
    count, seen = _READ_STREAK.get(path, (0, digest))
    count = count + 1 if seen == digest else 1     # changed on disk: start over
    _READ_STREAK[path] = (count, digest)
    if count < 3:
        return ""
    return (f"\n[you have read this file {count} times since anything was changed "
            f"or run, and it has not changed since the last read — make the edit, "
            f"or run the tests to see where things stand]")


def _norm(t: str) -> str:
    return "\n".join(ln.strip() for ln in t.strip().splitlines())


def _loose_find(text: str, needle: str) -> tuple[str, list[tuple[int, int]]]:
    """Spans matching ``needle`` once per-line whitespace is ignored."""
    want = _norm(needle)
    if not want:
        return want, []
    lines = text.splitlines(keepends=True)
    n = len(want.splitlines())
    spans, pos = [], []
    off = 0
    for ln in lines:
        pos.append(off)
        off += len(ln)
    for i in range(len(lines) - n + 1):
        chunk = "".join(lines[i:i + n])
        if _norm(chunk) == want:
            spans.append((pos[i], pos[i] + len(chunk.rstrip("\n"))))
    return want, spans


def _indent_of(block: str) -> str:
    for ln in block.splitlines():
        if ln.strip():
            return ln[:len(ln) - len(ln.lstrip())]
    return ""


def _reindent(matched: str, old: str, new: str) -> str:
    """Shift ``new`` by however much ``old`` was under-indented against the file."""
    have, want = _indent_of(old), _indent_of(matched)
    if have == want:
        return new
    out = []
    for ln in new.splitlines():
        if not ln.strip():
            out.append(ln)
        elif have and ln.startswith(have):
            out.append(want + ln[len(have):])
        elif not have:
            # No common prefix to strip: shift the whole block, keeping the
            # relative indentation inside it. Removing that instead is how the
            # first version flattened a nested body onto one level.
            out.append(want + ln)
        else:
            out.append(ln)
    return "\n".join(out) + ("\n" if new.endswith("\n") else "")


def _closest_region(text: str, needle: str) -> str:
    """The file's most similar block, so the model can correct rather than guess."""
    best, _, _ = _closest_region_at(text, needle)
    return best


def _closest_region_at(text: str, needle: str) -> tuple[str, float, int]:
    """(block, similarity, 1-based start line) of the file's closest match."""
    import difflib
    lines = text.splitlines()
    n = max(1, len(needle.splitlines()))
    best, score, at = "", 0.0, 0
    for i in range(0, max(1, len(lines) - n + 1)):
        chunk = "\n".join(lines[i:i + n])
        r = difflib.SequenceMatcher(None, chunk, needle).ratio()
        if r > score:
            best, score, at = chunk, r, i + 1
    return (best, score, at) if score > 0.5 else ("", score, 0)


def _line_numbers_of(text: str, needle: str) -> list[int]:
    out, start = [], 0
    while (i := text.find(needle, start)) != -1:
        out.append(text.count("\n", 0, i) + 1)
        start = i + 1
    return out


def _write_edit(f, updated: str) -> None:
    _push_undo(f)
    f.write_text(updated, encoding="utf-8")


def _append_hint(text: str, new: str) -> str:
    """Point at `append` when the edit was really an addition.

    The parameter existed and was never once used across five long-horizon
    runs: a description in a schema is not where a model looks. What it does
    read is the error in front of it. Measured repeatedly, and the reason
    scenario 2 kept hitting its turn cap: asked to add a test to a 1,700-line
    file, the model spent 25 Reads and 15 Greps against 4 Edits hunting for a
    unique anchor to hang the new code on.

    Only offered when it fits: the file is long enough that anchoring is the
    hard part, and the replacement is an addition rather than a rewrite of
    something already there.
    """
    if len(text.splitlines()) < 300 or len(new.strip()) < 40:
        return ""
    return ("\n\nIf you are adding new code rather than changing existing code, "
            "you do not need an anchor: call Write with append=true and just the "
            "new text.")


def _edit(p: dict, cfg: dict, tracker) -> str:
    path, old, new = p["file_path"], p["old_string"], p.get("new_string", "")
    f, why = _safe_target(path, cfg)
    if why:
        return f"Error: refused to edit {path} — {why}"
    if not f.exists():
        return f"Error: {path} does not exist"
    try:
        text = f.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading {path}: {e}"

    # The read-first rule stays, but refusing with nothing costs a whole turn to
    # learn something the harness already knows. Hand over the file with the
    # refusal so the next turn can be the edit itself.
    if tracker is not None and not tracker.has_read(str(f)):
        tracker.mark_read(str(f))
        return (f"Error: {path} had not been read yet, so the edit was not applied. "
                f"Here it is — reissue the same Edit if it is still right.\n\n"
                + _truncate("\n".join(f"{i + 1:6d}\t{ln}"
                                      for i, ln in enumerate(text.splitlines())),
                            output_cap(cfg)))

    if old not in text:
        # Small models reproduce a snippet with the indentation or inner spacing
        # slightly off. That is a rendering slip, not a different intent, so a
        # whitespace-insensitive match that is unique gets applied rather than
        # bounced — otherwise the model re-reads and guesses again, which is the
        # loop this harness exists to avoid. Anything ambiguous still refuses,
        # but shows the candidate so the next attempt can be exact.
        loose, where = _loose_find(text, old)
        if len(where) == 1 and not p.get("replace_all"):
            start, end = where[0]
            # Re-indent the replacement to the file's actual indentation.
            #
            # The model's snippet was under-indented — that is why the exact
            # match failed — and its new_string is under-indented by the same
            # amount. Substituting it verbatim silently reproduced the error in
            # the file, turning a refusal into corrupted Python. Applying the
            # match must not import the mistake that made it inexact.
            updated = text[:start] + _reindent(text[start:end], old, new) + text[end:]
            _write_edit(f, updated)
            return (f"Edited {path} (1 replacement; matched ignoring whitespace — "
                    f"your old_string differed only in spacing)")
        if where:
            first = text[where[0][0]:where[0][1]]
            return (f"Error: old_string not found in {path} exactly. {len(where)} "
                    f"near-match(es) differ only in whitespace. Did you mean "
                    f"this? Copy it exactly:\n\n"
                    + _truncate(first, output_cap(cfg))
                    + "\n\nOr pass replace_all=true to change every one.")
        # "Did you mean this?" — with the candidate in full.
        #
        # The hint used to be cut at 400 characters, which made it useless for
        # exactly the snippets that need it: the model cannot reproduce a block
        # it has only been shown part of, so a long old_string could never be
        # corrected and the model went round again. Watched live: two Edits
        # drifted further from the file each time (`p.get("offset", 0)` ->
        # `p["offset"]`, indent 4 -> 3 -> 2), then it gave up and overwrote the
        # module with a fragment.
        #
        # So the candidate is quoted whole, up to the ordinary output cap, with
        # the line it starts on. Copying it back is then a transcription the
        # model can actually perform.
        near, score, at = _closest_region_at(text, old)
        if near and score >= 0.6:
            return (f"Error: old_string not found in {path} — but line {at} is "
                    f"{score:.0%} similar. Did you mean this? Copy it exactly "
                    f"as old_string:\n\n"
                    + _truncate(near, output_cap(cfg))
                    + _append_hint(text, new))
        msg = (f"Error: old_string not found in {path}. Closest text in the "
               f"file:\n{_truncate(near, output_cap(cfg))}" if near else
               f"Error: old_string not found in {path}.")
        return msg + _append_hint(text, new)

    count = text.count(old)
    if count > 1 and not p.get("replace_all"):
        at = _line_numbers_of(text, old)
        return (f"Error: old_string appears {count} times in {path}, at lines "
                f"{', '.join(map(str, at[:12]))}. Add surrounding context to pick "
                f"one, or pass replace_all=true." + _append_hint(text, new))
    updated = text.replace(old, new) if p.get("replace_all") else text.replace(old, new, 1)
    try:
        _push_undo(f)
        f.write_text(updated, encoding="utf-8")
    except OSError as e:
        return f"Error writing {path}: {e}"
    return f"Edited {path} ({count if p.get('replace_all') else 1} replacement(s))"


# ── Bash ────────────────────────────────────────────────────────────────────
def _bash(p: dict, cfg: dict, tracker) -> str:
    cmd = p["command"]
    if (why := is_dangerous_bash(cmd)):
        return f"Error: refused — {why}"
    if (why := bash_outside_jail(cmd, cfg)):
        return f"Error: refused — {why}"
    timeout = min(int(p.get("timeout", 120) or 120), 600)
    try:
        watched = (watch_command(cmd, cfg, timeout)
                   if cfg.get("watch_bash", True) else None)
        if watched is not None:
            stdout, stderr, code, violation = watched
            if violation:
                return (f"Error: refused — the command touched {violation}, "
                        f"outside the working directory, and was stopped. "
                        f"Commands may only reach "
                        f"{', '.join(str(r) for r in jail_roots(cfg))}.")
            r = subprocess.CompletedProcess(cmd, code, stdout, stderr)
        else:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=cfg.get("_cwd") or os.getcwd(),
            )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error running command: {e}"
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
    if r.returncode != 0:
        out += f"\n[exit {r.returncode}]"
    # Notes go on after truncation, not before: a long test run is truncated
    # from the end, which is exactly where a note appended first would sit.
    notes = _test_regression(out, cfg) + _struggle(out, r.returncode)
    # Room is left for them, too: dispatch pages anything over the cap, and it
    # would cut the notes off the end of a result that fit only without them.
    # (_truncate's own marker is added on top of the limit it is given.)
    room = max(1000, output_cap(cfg) - len(notes) - 80)
    return _truncate(out.strip() or "[no output]", room) + notes


# Last test result seen per working directory. Session-scoped and deliberately
# not persisted: it exists to compare one run against the one before it.
_LAST_SUITE: dict[str, tuple[int, int]] = {}
_SUITE_RE = re.compile(
    r"(?:(?P<failed>\d+) failed)?(?:, )?(?:(?P<passed>\d+) passed)?"
    r"(?:, )?(?:(?P<error>\d+) error)?[^\n]*\bin [\d.]+s")


def _syntax_note(path: str, cfg: dict) -> str:
    """Say straight away when a write leaves a Python file that cannot parse.

    Watched across several live builds: a syntax or import error went unseen
    until the next test run, which then failed at collection and cost a whole
    cycle to find a typo the harness could have named on the spot. SWE-agent
    measured the same thing — checking an edit where it is made helps a model
    more than letting the test suite discover it. The write still happens;
    this only reports what the file now is.
    """
    if not path.endswith(".py"):
        return ""
    f = _resolve(path, cfg)
    try:
        source = f.read_text(encoding="utf-8", errors="replace")
        compile(source, str(f), "exec")
    except SyntaxError as e:
        line = (e.text or "").rstrip()
        where = f"line {e.lineno}" + (f": `{line.strip()[:120]}`" if line.strip() else "")
        return (f"\n[this leaves a syntax error at {where} — {e.msg}. The file was "
                f"written as given; fix this before running anything]")
    except (OSError, ValueError):
        return ""
    return ""


def _test_regression(out: str, cfg: dict) -> str:
    """Say when a test run came back worse than the one before it.

    The harness runs the suite, reads the counts, and used to keep them to
    itself. Watched live on a build task: the model rewrote a module, went from
    1 failing test to 3, and carried on — it had no reason to re-read a number
    it had already seen scroll past. The comparison is free and the harness is
    the only party holding both halves of it.

    A statement, not a veto. Breaking tests on the way to a bigger change is
    ordinary; not noticing is the problem.
    """
    m = None
    # Only short lines near the end. The pattern's `[^\n]*` rescans the rest of
    # the line from every starting position, so on one long line it was
    # quadratic: 10 s for 60,000 characters, minutes for a minified file. A
    # test summary is a short line at the end of a run.
    tail = "\n".join(l for l in out[-40000:].splitlines() if len(l) <= 500)
    for m2 in _SUITE_RE.finditer(tail):
        if m2.group("passed") or m2.group("failed"):
            m = m2                       # the last summary line wins
    if not m:
        return ""
    failed = int(m.group("failed") or 0) + int(m.group("error") or 0)
    passed = int(m.group("passed") or 0)
    key = str(cfg.get("_cwd") or os.getcwd())
    before = _LAST_SUITE.get(key)
    _LAST_SUITE[key] = (failed, passed)
    if before is None:
        return ""
    was_failed, was_passed = before
    if failed > was_failed or passed < was_passed:
        return (f"\n[the suite got worse: {failed} failed / {passed} passed, "
                f"was {was_failed} failed / {was_passed} passed. "
                f"`git diff` shows what changed since the last commit.]")
    if failed < was_failed and failed == 0:
        return f"\n[the suite is green again: {passed} passed]"
    return ""


# ── Hints when the same failure keeps coming back ───────────────────────────
# Within one turn: the failure most recently seen, and how many times running.
_FAIL_STREAK: dict[str, object] = {"sig": "", "count": 0, "told": 0, "edited": False}
# Bounded: an unbounded \w+ ahead of "Error" backtracks at every position of a
# long line, which is quadratic — a 200,000-character line of minified output
# took minutes of CPU. Exception names are short, and only line tails are read.
_FAIL_LINE = re.compile(r"(\b\w{1,64}(?:Error|Exception)\b.*|FAILED .*|assert .*|Error: .*)")


def new_turn() -> None:
    """A new message from the user: hints start over."""
    _FAIL_STREAK.update(sig="", count=0, told=0, edited=False)


def struggle_level() -> int:
    """0, or the highest hint given this turn (1-3). The loop surfaces 3."""
    return int(_FAIL_STREAK["told"])


def _failure_signature(out: str) -> str:
    # The verdict is at the end of a run; look at the last lines, and only at
    # the last 400 characters of each, so the cost is flat however long the
    # output is.
    lines = [l.strip()[-400:] for l in out[-40000:].splitlines()
             if l.strip() and not l.startswith("[exit")][-60:]
    # A test run's pass/fail counts are part of the signature. The failing
    # line alone reads the same while the model fixes other tests one by one
    # — the last test in the file keeps failing — and on a live run that
    # said "not converging" to a model that had just gone from 10 passing to
    # 13. Fixing a test is progress, and progress starts the count over.
    tail = "\n".join(l for l in lines if len(l) <= 500)
    counts = ""
    for m in _SUITE_RE.finditer(tail):
        if m.group("passed") or m.group("failed"):
            counts = f"{m.group('failed') or 0} failed, {m.group('passed') or 0} passed · "
    for line in reversed(lines):
        if (m := _FAIL_LINE.search(line)):
            return (counts + m.group(1))[:240]
    return (counts + lines[-1])[:240] if lines else ""


def _struggle(out: str, returncode: int) -> str:
    """Escalating hints when a run keeps ending in the same failure.

    Like a game's hints, and for the same reason: the harness cannot know the
    answer, but it can know for certain that the player is stuck. Watched on
    two live builds: `AssertionError: a.grad = 1.0` returned run after run for
    twenty minutes, while a working WebSearch tool sat unused — zero searches
    in either run. The model was not short of attempts; it was short of a
    different approach, and nothing told it so.

    So the hints escalate the *strategy*, never the answer: step back and
    restate the gap; then go and look; then stop and ask. Each is attached to
    the result that triggered it, never sent as a user message of its own
    (§3.3.1), and a different failure — which is progress — starts over.
    """
    edited = bool(_FAIL_STREAK["edited"])
    _FAIL_STREAK["edited"] = False
    if returncode == 0:
        # A command that "succeeds" can be just as stuck. Watched live: the
        # model probed with `python3 -c` scripts that printed "Expected:
        # a.grad = 2.0" and exited 0, seven times, editing the code between
        # each — and no hint fired, because only failures were counted. The
        # signal is the code changing while the result does not. Without an
        # edit in between, an unchanged result means nothing (`ls` twice).
        tail = [l.strip() for l in out.splitlines()
                if l.strip() and not l.startswith("[exit")][-3:]
        sig = " / ".join(tail)[:200]
        if not sig or not edited or sig != _FAIL_STREAK["sig"]:
            _FAIL_STREAK.update(sig=sig, count=1 if edited else 0)
            return ""
        _FAIL_STREAK["count"] = int(_FAIL_STREAK["count"]) + 1
    else:
        sig = _failure_signature(out)
        if not sig:
            return ""
        if sig == _FAIL_STREAK["sig"]:
            _FAIL_STREAK["count"] = int(_FAIL_STREAK["count"]) + 1
        else:
            _FAIL_STREAK.update(sig=sig, count=1)
    n = int(_FAIL_STREAK["count"])
    if n == 3:
        _FAIL_STREAK["told"] = max(1, int(_FAIL_STREAK["told"]))
        return (f"\n[hint: this is the 3rd run in a row ending in `{sig}`. Before "
                f"editing again, write one line saying what the check "
                f"expects and what your code does instead — the gap between those "
                f"two is the bug.]")
    if n == 5:
        _FAIL_STREAK["told"] = max(2, int(_FAIL_STREAK["told"]))
        return (f"\n[hint: 5 runs in a row now end in `{sig}`, so the current "
                f"approach is not converging. Look up how this is normally done "
                f"before trying again: WebSearch for the concept (not your code), "
                f"then WebFetch a page that explains it.]")
    if n == 7:
        _FAIL_STREAK["told"] = 3
        return (f"\n[hint: 7 runs in a row end in `{sig}`. Stop trying variations. "
                f"Report to the user what you have tried, what you think is wrong, "
                f"and what you would need to know to fix it.]")
    return ""


# ── Glob / Grep ─────────────────────────────────────────────────────────────
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
              "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", "target"}


def _walk(root: Path):
    # A single file is a legitimate search root.
    #
    # os.walk yields nothing for a file, so `Grep path=some/file.py` returned
    # "No matches" while the same search one directory up returned nine — and
    # the harness's own hint tells the model to narrow with `path`. Watched
    # live: it narrowed as instructed, was told there was nothing there, and
    # spent the next several calls doubting a result it had already found.
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            yield Path(dirpath) / fn


def _glob_match(rel: str, name: str, pattern: str) -> bool:
    """fnmatch, plus the `**/` case fnmatch gets wrong.

    fnmatch's `*` already crosses separators, so `**/*.py` matches `src/a.py` —
    but it requires the literal `/`, so it silently misses `a.py` at the root.
    `**/*.py` is the single most common glob a model writes, and a wrong-but-
    empty result reads as "no such files" rather than "bad pattern".
    """
    if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
        return True
    if pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:]):
        return True
    return False


def _glob(p: dict, cfg: dict, tracker) -> str:
    root, why = _safe_target(str(p.get("path") or cfg.get("_cwd") or os.getcwd()), cfg)
    if why:
        return f"Error: refused to search {p.get('path')} — {why}"
    pattern = p["pattern"]
    hits = []
    for f in _walk(root):
        rel = str(f.relative_to(root))
        if _glob_match(rel, f.name, pattern):
            hits.append((f.stat().st_mtime, rel))
    if not hits:
        return f"No files matching {pattern!r} under {root}"
    hits.sort(reverse=True)  # most recently modified first
    return _truncate("\n".join(r for _, r in hits[:200]), output_cap(cfg))


_TEST_HINT = re.compile(r"(^|/)(tests?|spec)s?/|(^|/)test_|_test\.[a-z]+$")


def _rank_matches(raw: str, root) -> str:
    """Group matches by file, implementation before tests, and say how many.

    Search output arrives in filesystem walk order, so a question about what the
    code does can return three test files before the source. Watched live: the
    model read the first few groups, found test assertions and docstrings,
    concluded the search had missed, and retried with a trivially different
    pattern — twice. The ordering was arbitrary, so there was no rule it could
    have learned.

    Tests are still included: sometimes the test *is* the answer. They are just
    no longer in the way of the implementation.
    """
    if not raw:
        return raw
    groups: dict[str, list[str]] = {}
    for line in raw.splitlines():
        if line == "--":
            continue
        path = line.split(":", 1)[0].split("-", 1)[0]
        groups.setdefault(path, []).append(line)

    def hits(lines: list[str]) -> int:
        # ripgrep separates a match with ':' and context with '-'
        return sum(1 for ln in lines if re.match(r"^[^:]+:\d+:", ln))

    ordered = sorted(groups.items(),
                     key=lambda kv: (bool(_TEST_HINT.search(kv[0])), -hits(kv[1]), kv[0]))
    out = []
    for path, lines in ordered:
        n = hits(lines)
        label = " (tests)" if _TEST_HINT.search(path) else ""
        out.append(f"{path}{label} — {n} match{'es' if n != 1 else ''}")
        out.extend("  " + ln for ln in lines)
        out.append("")
    return "\n".join(out).strip()


def _no_match_help(pattern: str, root, cfg: dict) -> str:
    """A search that found nothing should still move the model forward.

    Requiring a *particular* tool for a common need is a design failure: across
    ten runs on a real repository the model reached for Grep 182 times and
    FindSymbol four, then spent fifteen consecutive calls hunting a definition
    whose name differed from the tool's by an underscore. Rather than telling it
    to use the other tool, Grep does the work: strip the definition keyword,
    try the obvious name variants, and report what actually exists.
    """
    from .repomap import find_symbol

    # A definition-shaped query — `def foo`, `class Bar` — is really a search
    # for the name. Try the name on its own.
    bare = re.sub(r"^\^?\s*(?:def|class|func|fn|function|struct|type|interface"
                  r"|impl|var|const)\s+", "", pattern).strip()
    bare = bare.strip("^$").strip()
    if bare and bare != pattern and re.fullmatch(r"[\w.]+", bare):
        hit = find_symbol(str(root), bare)
        if not hit.startswith("No definition"):
            return (f"No matches for {pattern!r}, but that name exists — "
                    f"dropping the keyword finds it:\n\n{hit}")
        if "Closest matches" in hit:
            return f"No matches for {pattern!r}. {hit}"
    if re.fullmatch(r"[\w.]+", pattern.strip("^$")):
        hit = find_symbol(str(root), pattern.strip("^$"))
        if "Closest matches" in hit or not hit.startswith("No definition"):
            return f"No matches for {pattern!r}. {hit}"
    return (f"No matches for {pattern!r}. Try a shorter or partial term, drop "
            f"`def`/`class` from the pattern, set case_insensitive, or widen "
            f"`path`.")


def _grep(p: dict, cfg: dict, tracker) -> str:
    root, why = _safe_target(str(p.get("path") or cfg.get("_cwd") or os.getcwd()), cfg)
    if why:
        return f"Error: refused to search {p.get('path')} — {why}"
    pattern = p["pattern"]
    glob_filter = p.get("glob")
    flags = re.IGNORECASE if p.get("case_insensitive") else 0
    # ripgrep when available — it's faster and respects .gitignore.
    if shutil.which("rg"):
        # Context lines, not just the matching line.
        #
        # Returning `path:line:match` alone tells the model *where* something is
        # and not *what it says*, so it has to guess an offset for a follow-up
        # Read. Watched live on a real repository: Grep found the answer twice,
        # the model read the wrong region of the file, and went back to grepping
        # until it ran out of turns. A few lines either side usually answers the
        # question outright.
        ctx = max(0, min(int(p.get("context", 3) or 0), 10))
        cmd = ["rg", "--line-number", "--no-heading", "--color=never", "-m", "50"]
        if ctx:
            cmd += ["-C", str(ctx)]
        if flags:
            cmd.append("-i")
        if glob_filter:
            cmd += ["--glob", glob_filter]
        cmd += ["-e", pattern, str(root)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            out = _rank_matches(r.stdout.strip(), root)
            if out:
                return _truncate(out, output_cap(cfg))
            return _no_match_help(pattern, root, cfg)
        except Exception:
            pass  # fall through to the pure-Python path
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: bad regex {pattern!r}: {e}"
    ctx = max(0, min(int(p.get("context", 3) or 0), 10))
    lines: list[str] = []
    for f in _walk(root):
        if glob_filter and not fnmatch.fnmatch(f.name, glob_filter):
            continue
        try:
            body = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        try:
            rel = f.relative_to(root) if root.is_dir() else Path(f.name)
        except ValueError:
            rel = f
        for i, ln in enumerate(body, 1):
            if not rx.search(ln):
                continue
            lo, hi = max(0, i - 1 - ctx), min(len(body), i + ctx)
            for j in range(lo, hi):
                sep = ":" if j == i - 1 else "-"
                lines.append(f"{rel}:{j + 1}{sep}{body[j]}")
            if ctx:
                lines.append("--")
            if len(lines) > 400:
                break
        if len(lines) > 400:
            break
    out = _rank_matches("\n".join(lines).strip(), root)
    if out:
        return _truncate(out, output_cap(cfg))
    return _no_match_help(pattern, root, cfg)



# ── WebFetch ────────────────────────────────────────────────────────────────
def _webfetch(p: dict, cfg: dict, tracker) -> str:
    from . import net as requests
    url = p["url"]
    if not url.startswith(("http://", "https://")):
        return "Error: url must start with http:// or https://"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "miniharness/0.1"})
        r.raise_for_status()
    except Exception as e:
        return f"Error fetching {url}: {e}"
    text = r.text
    if "html" in r.headers.get("Content-Type", ""):
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;?", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return _truncate(text.strip(), 20_000)


# ── WebSearch ───────────────────────────────────────────────────────────────
def _websearch(p: dict, cfg: dict, tracker) -> str:
    """Search the web. Titles, URLs and snippets; WebFetch reads a page in full.

    Cut originally on the argument that coding turns should not be tempted to
    search. That reasoning came from frontier models, which already know what
    `CMake Error: could not find Minisat` implies. A small model does not, and
    without search it has no route from an unfamiliar error to the fix — the
    dead end seen repeatedly in real runs. The research sub-loop already showed
    the model uses search well when given it: search, fetch, extract, cite.
    """
    from .research import _ddg_search
    q = str(p.get("query") or "").strip()
    if not q:
        return "Error: query is required"
    n = max(1, min(int(p.get("max_results", 5) or 5), 8))
    out = _ddg_search(q, n)
    return out.strip() or "No results. Try different terms."


# ── Schemas ─────────────────────────────────────────────────────────────────
# ORDER IS PART OF THE PROMPT PREFIX. Appending is safe; reordering or inserting
# invalidates every cached prefix on every machine (DESIGN.md §5.1).
SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "Read",
        "description": "Read a file. Returns numbered lines. Read a file before editing it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute or relative path"},
                "offset": {"type": "integer", "description": "First line to read (0-based)"},
                "limit": {"type": "integer", "description": "Max lines (default 2000)"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": ("Write a file, creating or overwriting it. Read an "
                        "existing file first. Use append to add to the end of "
                        "a large file instead of hunting for an Edit anchor."),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
                # Appended to the schema, never inserted: property order is part
                # of the prompt prefix, and reordering it invalidates every
                # cached prefix (DESIGN.md 5.1).
                "append": {"type": "boolean"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": (
            "Replace an exact string in a file. old_string must be unique unless "
            "replace_all is set."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Bash",
        "description": "Run a shell command and return its output. Use for tests, git, builds.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds, max 600"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Glob",
        "description": "Find files by name pattern (e.g. '**/*.py'). Newest first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Directory to search (default cwd)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents by regex. Returns path:line:match.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string", "description": "Filter filenames, e.g. '*.py'"},
                "case_insensitive": {"type": "boolean"},
                "context": {"type": "integer",
                            "description": "Lines of context around each match "
                                           "(default 3, max 10)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "WebFetch",
        "description": "Fetch a URL and return its text content with HTML stripped.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        # Appended at the end on purpose: schema order is part of the prompt
        # prefix, so adding here is safe while inserting anywhere else would
        # invalidate every cached prefix (DESIGN.md 5.1).
        "name": "WebSearch",
        "description": "Search the web for documentation, error messages, or "
                       "library behaviour. Returns titles, URLs and snippets; "
                       "use WebFetch to read a result in full.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "description": "1-8, default 5"},
            },
            "required": ["query"],
        },
    },
]

_IMPLS = {
    "Read": _read, "Write": _write, "Edit": _edit, "Bash": _bash,
    "Glob": _glob, "Grep": _grep,
    "WebFetch": _webfetch,
    "WebSearch": _websearch,
}

# Tools that change something on disk. The permission flow asks about these;
# everything else is read-only and runs unprompted.
MUTATING = {"Write", "Edit", "Bash"}


def schemas_for(config: dict) -> list[dict]:
    """The tool schemas to advertise, in fixed order."""
    return list(SCHEMAS)


# Recent read-only results, keyed by (tool, normalised arguments). Serving a
# repeat from here costs nothing and always hands the model something it can
# use. Bounded because it is a cache, not a log.
_RECENT: "OrderedDict[tuple[str, str], str]" = OrderedDict()
_RECENT_MAX = 24


def dispatch(name: str, params: dict, config: dict, tracker=None,
             continue_from: int = 0) -> str:
    """Run a tool. Never raises — a tool error is a message to the model.

    Output larger than the window's share is served in pieces rather than cut
    off. ``continue_from`` is where the last piece ended, and the loop supplies
    it when the model asks for the same thing again — so "give me that again"
    means "give me the rest", the same bargain continuation makes for a reply
    that outgrew its cap. Nothing is withheld; it just arrives across turns.
    """
    impl = _IMPLS.get(name)
    if impl is None:
        return f"Error: unknown tool {name!r}. Available: {', '.join(_IMPLS)}"
    note_other_tool_use(name)
    if "_raw" in params:
        return "Error: arguments were malformed JSON (likely truncated). Re-issue the call."
    if "_stopped" in params:
        where = f" for {params['file_path']}" if params.get("file_path") else ""
        return (f"Error: this {name} call{where} was stopped while it was being "
                f"written, after {int(params.get('_chars', 0)):,} characters — "
                f"{params['_stopped']}. Nothing was run or written. Look at what "
                f"you were repeating before trying again; to write a long file, "
                f"write the first part, then add the rest with append=true.")
    try:
        out = impl(params, config, tracker)
    except KeyError as e:
        return f"Error: {name} missing required parameter {e}"
    except Exception as e:  # a crashing tool must not kill the loop
        return f"Error: {name} failed: {type(e).__name__}: {e}"
    if name in ("Write", "Edit") and not out.startswith("Error"):
        out += _syntax_note(str(params.get("file_path", "")), config)
    page = _paginate(name, out, config, continue_from)
    if name == "Read" and not page.startswith("Error") and params.get("file_path"):
        page += _reread_streak(str(_resolve(str(params["file_path"]), config)))
    return page


def _paginate(name: str, out: str, config: dict, start: int) -> str:
    cap = output_cap(config)
    base = 0
    if start:
        rest_of = out[start:]
        trimmed = rest_of.lstrip("\n")
        # Track what the trim removed: the resume offset must stay in the
        # original string's coordinates or it slips by a newline every round,
        # which walks every second piece into the middle of a line.
        base = start + (len(rest_of) - len(trimmed))
        out = trimmed
        if not out.strip():
            # Carry the bookmark forward, or this message is not sticky: the
            # next identical call would find no resume point and start the
            # whole file again.
            return (f"[no further output from {name}; you have all of it "
                    f"(resume={base})]")
    if len(out) <= cap:
        if not start:
            return out
        # The last piece must say that it is the last piece.
        #
        # Without a bookmark here the next identical call finds no resume point
        # and starts the file again from the top. Measured after duplicate
        # suppression was removed — which had been hiding this — a Read walked
        # 7,920 -> 47,525 and then restarted at 7,920 and looped. This is not a
        # refusal: the model gets told where it is, which is what lets it stop.
        return (f"[continued]\n{out}\n\n[end of {name} output "
                f"(resume={base + len(out)})]")
    # Cut at a line boundary. Splitting mid-line hands back a chunk with no
    # line number and half an identifier, which is far harder to use than one
    # slightly shorter piece.
    cut = out.rfind("\n", 0, cap)
    if cut < cap // 2:
        cut = cap                      # a single line longer than the budget
    piece, rest = out[:cut], len(out) - cut
    head = "[continued]\n" if start else ""
    if name in MUTATING:
        # Repeating a Bash call would re-run the command, which is not a way to
        # read more of anything — so point at a way that is.
        more = (f"\n\n[{rest} more characters not shown. Re-run narrowed, e.g. "
                f"piped through `tail -c +{cap}`, `sed -n`, or `grep`.]")
    else:
        # The resume point rides in the message itself, so it cannot drift
        # from what was actually sent, and it disappears exactly when
        # compaction removes the piece it refers to — at which point starting
        # over is the correct behaviour.
        more = (f"\n\n[{rest} more characters not shown. Repeat this exact "
                f"{name} call to continue. (resume={base + cut})]")
    return head + piece + more
