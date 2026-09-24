"""The repository map: what exists, and which of it matters right now.

Rewritten from zero. The previous version was 661 lines of vendored Aider —
tree-sitter parsing, a symbol reference graph, personalized PageRank, five
hand-tuned edge multipliers, and a sqlite tag cache — plus six dependencies
(``tree-sitter``, ``grep-ast``, ``networkx``, ``diskcache``, ``pygments``,
``tree-sitter-language-pack``).

What replaced it is this file, and the reason is a measurement. Comparing the
map on and off across seven runs on a 295-file repo, the median tool calls to
locate code fell from 8 to 2 — but **the answer was never in the map**. The
symbol it was asked to find did not appear in the rendered output at all. What
the map supplied was *orientation*: enough of the repo's shape and naming
conventions that the model wrote a good `Grep` on the first try instead of
guessing filenames.

Paths already carry that. A path states the structure (directories are
subsystems), the naming convention (``tool_call_recovery.py`` tells you a great
deal), and the scale. Parsing every file to extract symbols, ranking them by
PageRank over a graph of shared identifiers, and then rendering a token-budgeted
subset was an expensive way to deliver information the directory listing already
contains.

So: no parsing, no graph, no cache, no dependencies. List the files, rank them
against the task, render what fits.

**The symbol listing is redundant, not merely expensive.** Tested on the most
adversarial tasks available — four symbols whose filename shares no word with
them, so no path map could reveal the answer — every arm succeeded 4/4,
*including with no map at all*, because ``FindSymbol`` is a grep and answers
"where is X defined" in one tool call. Adding symbols to the map saved half a
tool call per task and was the **slowest** arm (5.8 s vs 4.6 s): the larger
prompt costs more to process every turn than the occasional tool call it spares.
Symbol lookup is precisely the case where paying on demand beats paying always.
"""

from __future__ import annotations

import os
import re
import subprocess

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]+")
# Directories that are never source, whether or not git is asked.
_SKIP = {"node_modules", "__pycache__", "venv", ".venv", "build", "dist",
         "target", "vendor", "third_party"}


def list_files(root: str) -> list[str]:
    """Repo-relative source files.

    Asks git first. A repository already tracks exactly the files that are
    source and ignores exactly the ones that are not, so parsing ``.gitignore``
    ourselves — which an earlier version did in fifty lines, with negation and
    anchoring rules — is reimplementing something already installed.
    """
    try:
        # --others --exclude-standard: files the agent has created but nobody
        # has committed yet are source too. Tracked files alone left a new
        # project's map showing only what was in the first commit — watched:
        # after a compaction the agent read the map, saw a single markdown
        # file, and concluded the two modules it had just written were absent.
        out = subprocess.run(["git", "-C", root, "ls-files", "--cached",
                              "--others", "--exclude-standard"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            seen: set[str] = set()
            files = []
            for ln in out.stdout.splitlines():
                parts = ln.split("/")
                # A repository with no .gitignore still has caches in it.
                if (not ln or ln in seen
                        or any(d in _SKIP or d.startswith(".") for d in parts[:-1])
                        or ln.endswith((".pyc", ".pyo"))
                        # --cached still lists a tracked file that was deleted.
                        or not os.path.isfile(os.path.join(root, ln))):
                    continue
                seen.add(ln)
                files.append(ln)
            return files
    except (OSError, subprocess.SubprocessError):
        pass

    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in _SKIP and not d.startswith(".")]
        for f in filenames:
            if not f.startswith("."):
                files.append(os.path.relpath(os.path.join(dirpath, f), root))
    return files


def words_in(text: str) -> set[str]:
    """Lowercase word tokens, splitting snake_case and camelCase alike."""
    return {w.lower() for w in _WORD.findall(re.sub(r"[_\-./]", " ", text))}


def score(path: str, task: set[str], open_files: set[str]) -> float:
    """How much this file matters to the task at hand.

    Three signals, in the order a person would use them: it is already open,
    its name shares words with the request, it is shallow (top-level files are
    entry points, deeply nested ones are details).
    """
    if path in open_files:
        return 100.0
    shared = len(words_in(path) & task)
    depth = path.count("/")
    return shared * 10.0 + 1.0 / (1.0 + depth)


def render(root: str, task_text: str = "", open_files=(), budget_chars: int = 3000) -> str:
    """A map of the repo, most relevant first, truncated to a character budget.

    Grouped by directory because that is how the structure reads, and because a
    directory with forty files should cost one line, not forty.
    """
    files = list_files(root)
    if not files:
        return ""
    task = words_in(task_text)
    opened = {f for f in open_files}

    ranked = sorted(files, key=lambda f: (-score(f, task, opened), f))

    by_dir: dict[str, list[str]] = {}
    for f in ranked:
        by_dir.setdefault(os.path.dirname(f) or ".", []).append(f)

    # Directory order follows its best file, so relevance drives the layout.
    best = {d: min(ranked.index(f) for f in fs) for d, fs in by_dir.items()}
    # Anchor the map to a real directory, and show the form a path should take.
    #
    # The map groups by directory ("tests/  test_core.py  ..."), which the model
    # read as a top-level root: it asked for `/tests/test_core.py`, then
    # `/workspace/tests/test_core.py`, and said so in its reasoning — "let me
    # use the absolute path from the repository map". The jail refused every
    # one, correctly. It ran `pwd` and `ls -la`, saw the truth, and went back to
    # `/workspace` anyway: 12 of its first 17 calls were spent on a path that
    # could never resolve. Nothing had ever told it where it was.
    lines = [f"Repository map ({len(files)} files), most relevant first.",
             f"Working directory is {root} — paths below are relative to it, so "
             f"read `tests/test_core.py`, never `/tests/test_core.py`:"]
    used = sum(len(l) for l in lines)
    # Counted, not derived from len(lines): the remaining-directory count used
    # to be `len(by_dir) - len(lines) + 1`, which silently assumed exactly one
    # header line and went wrong the moment a second was added.
    written = 0
    for d in sorted(by_dir, key=lambda d: best[d]):
        names = [os.path.basename(f) for f in by_dir[d]]
        head = f"  {d}/" if d != "." else "  ."
        shown, n = [], 0
        for nm in names:
            if n + len(nm) > 220:
                shown.append(f"...+{len(names) - len(shown)} more")
                break
            shown.append(nm)
            n += len(nm) + 2
        line = f"{head}  " + "  ".join(shown)
        if used + len(line) > budget_chars:
            lines.append(f"  ...and {len(by_dir) - written} more directories")
            break
        lines.append(line)
        written += 1
        used += len(line)
    return "\n".join(lines) + "\n"


# ── the two tools ───────────────────────────────────────────────────────────
HAVE_GRAPH = True          # no optional extra any more; this always works


def repo_map(root: str, focus=(), max_tokens: int = 800, mentions=()) -> str:
    return render(root, " ".join(mentions), focus, budget_chars=max_tokens * 4)


def find_symbol(root: str, name: str) -> str:
    """Locate a definition by name.

    Grep with a definition-shaped pattern finds this in every language without
    a parser, a grammar, or a language pack. The previous implementation used
    tree-sitter queries maintained per language, of which two (C and C++) had no
    reference captures at all and silently fell back to a lexer that emitted
    every identifier in the file.
    """
    pat = (rf"(def|class|func|fn|function|struct|type|interface|impl|var|const)"
           rf"\s+{re.escape(name)}\b|^\s*{re.escape(name)}\s*[:=]")
    try:
        out = subprocess.run(
            ["grep", "-rnE", "--exclude-dir=.git", pat, root],
            capture_output=True, text=True, timeout=30)
        hits = [ln for ln in out.stdout.splitlines() if ln][:20]
    except (OSError, subprocess.SubprocessError):
        hits = []
    if not hits:
        # Try the obvious near-misses before giving up.
        #
        # A model asked to change the Read tool searches for the tool's name,
        # while the implementation carries a leading underscore and lower case —
        # one character and a case away. Watched live: fifteen consecutive calls
        # hunting a definition that could never match, because nothing bridged
        # the tool's name and the function's. Failing with "no definition found"
        # invites exactly that loop; naming the near-misses ends it.
        variants = []
        for cand in (f"_{name}", f"_{name.lower()}", name.lower(),
                     f"_{name}_", name.upper()):
            if cand == name or cand in variants:
                continue
            variants.append(cand)
        found: list[tuple[str, list[str]]] = []
        for cand in variants:
            vpat = (rf"(def|class|func|fn|function|struct|type|interface|impl|"
                    rf"var|const)\s+{re.escape(cand)}\b")
            try:
                vout = subprocess.run(
                    ["grep", "-rnE", "--exclude-dir=.git", vpat, root],
                    capture_output=True, text=True, timeout=15)
                vhits = [ln for ln in vout.stdout.splitlines() if ln][:5]
            except (OSError, subprocess.SubprocessError):
                vhits = []
            if vhits:
                found.append((cand, vhits))
        if found:
            parts = [f"No definition named exactly {name!r}. Closest matches:"]
            for cand, vhits in found[:3]:
                parts.append(f"\n{cand}:")
                parts += ["  " + h.replace(root.rstrip("/") + "/", "") for h in vhits]
            return "\n".join(parts)
        return (f"No definition of {name!r} found, and no close variant either "
                f"(tried {', '.join(variants[:4])}). It may be spelled "
                f"differently, or defined in a table rather than declared — try "
                f"Grep for the bare word without `def`.")
    body = "\n".join(h.replace(root.rstrip("/") + "/", "") for h in hits)
    return f"# Definition(s) of {name}\n{body}"


def find_src_files(directory: str) -> list[str]:
    """Absolute paths, kept for context.fingerprint()."""
    return [os.path.join(directory, f) for f in list_files(directory)]
