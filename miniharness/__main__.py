"""CLI and REPL.

Seven slash commands, one file. Promethean had eleven command modules totalling
5,848 lines; almost all of it was surface for features that no longer exist.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time as _time

from rich.console import Console
from rich.markup import escape
from rich.text import Text

from . import config as cfg_mod
from . import checkpoint, context, loop, models, preview, research, server, session, tools

class Transcript:
    """Everything shown on screen, as plain text, in a file beside the session.

    The terminal is the only place a run could be seen, and a terminal cannot
    be read from outside it: during a runaway, all that was visible elsewhere
    was the server's token count, never what the model was writing. The model's
    reasoning goes here too, although the screen only shows a counter for it,
    because that is exactly what is needed to tell hard work from circling.

    Line-buffered and flushed on every write, so `tail -f` shows a run as it
    happens. Never raises: losing a log line must not take down a turn.
    """

    def __init__(self):
        self.path: pathlib.Path | None = None
        self._file = None
        self._console: Console | None = None

    def open(self, session_id: str) -> None:
        path = session.SESSIONS / f"{session_id}.log"
        if path == self.path:
            return
        self.close()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", encoding="utf-8", buffering=1)
        except OSError:
            return
        self.path = path
        # No colour, no terminal: plain text a person or a script can read.
        self._console = Console(file=self._file, width=200, color_system=None,
                                force_terminal=False, highlight=False)
        self.raw(f"\n=== {_time.strftime('%Y-%m-%d %H:%M:%S')} · session "
                 f"{session_id} ===\n")

    def print(self, *args, **kwargs) -> None:
        if self._console is None:
            return
        try:
            self._console.print(*args, **kwargs)
            self._file.flush()
        except Exception:
            pass

    def raw(self, text: str) -> None:
        if self._file is None:
            return
        try:
            self._file.write(text)
            self._file.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
        self.path = self._file = self._console = None


TRANSCRIPT = Transcript()


class _TeeConsole:
    """The screen's console, with every print and every answer also sent to
    the transcript. Everything else — status lines, the pager, the size —
    is the real console's, unchanged."""

    def __init__(self, real: Console):
        self._real = real

    def print(self, *args, **kwargs) -> None:
        self._real.print(*args, **kwargs)
        TRANSCRIPT.print(*args, **kwargs)

    def input(self, prompt: str = "", **kwargs) -> str:
        answer = self._real.input(prompt, **kwargs)
        TRANSCRIPT.print(prompt, end="")
        TRANSCRIPT.raw(answer + "\n")
        return answer

    def __getattr__(self, name):
        return getattr(self._real, name)


console = _TeeConsole(Console())

HELP = """\
[bold]Commands[/bold]
  /help                 this list
  /model [name]         show or switch model (e.g. gpt-4o, ollama/qwen3, local)
  /models               pick and download a local model for this machine
  /config [k=v]         show or set configuration
  /compact              shrink the conversation now
  /think [on|off|save]  show the last turn's reasoning, toggle live display,
                        or save it to a file
  /diff [-n]            what changed this session (or since n changes ago)
  /undo                 revert the last file write or edit
  /checkpoints          list the snapshots taken after each accepted change
  /rewind [-n|sha]      restore the working tree to a checkpoint
  /clear                start a fresh conversation (files are left as they are)
  /resume [id]          resume a previous session (default: the most recent)
  /log                  where this session's plain-text log is
  /research <question>  spawn a bounded research run; writes a markdown report
  /quit                 exit

[bold]In a message[/bold]
  @path                 attach a file — the model sees it without spending a turn
  !command              run a shell command yourself; its output goes with your
                        next message
  Tab                   complete /commands and @paths

[dim]Ctrl-C interrupts a turn. Ctrl-D exits.[/dim]"""


# ── First-run model setup ───────────────────────────────────────────────────
def choose_model(config: dict) -> bool:
    """Detect hardware, show the menu, download. True if a model got configured."""
    hw = models.detect_hardware()
    budget = hw.budget_gb
    if not budget:
        console.print("[yellow]Could not detect VRAM or RAM.[/yellow] "
                      "Set a model manually: /config llama_model_path=/path/to.gguf")
        return False

    where = f"{hw.vram_gb:.1f} GB VRAM" if hw.vram_gb else f"{hw.ram_gb:.0f} GB RAM (CPU)"
    if hw.gpu_name:
        where += f" ({hw.gpu_name})"
    console.print(f"Detected: [bold]{where}[/bold]\n")

    with console.status("Fetching available quantizations from Hugging Face..."):
        menu = models.build_menu(budget)
    if not menu:
        console.print("[yellow]No model in the catalog fits, or Hugging Face is "
                      "unreachable.[/yellow]")
        return False

    for i, pick in enumerate(menu, 1):
        tag = " [green](recommended)[/green]" if pick.recommended else ""
        # Say when a context is short of what the model supports, and which KV
        # type buys it. Without this the menu offers 151K and 256K side by side
        # with no indication that one of them is a crippled model.
        if pick.fits_target:
            ctx = f"{pick.ctx_k:>4}K ctx (full) at {pick.kv_quant} KV"
        else:
            ctx = (f"{pick.ctx_k:>4}K ctx [yellow]of {pick.model.max_ctx_k}K"
                   f"[/yellow] at {pick.kv_quant} KV")
        console.print(
            f"  [bold]{i}.[/bold] {pick.model.key:<16} {pick.quant.label:<10} "
            f"{pick.quant.size_gb:>5.1f} GB  ->  {ctx}{tag}"
        )
    console.print()

    try:
        raw = console.input("Pick [1]: ").strip() or "1"
        choice = menu[int(raw) - 1]
    except (ValueError, IndexError):
        console.print("[red]Invalid choice.[/red]")
        return False
    except (EOFError, KeyboardInterrupt):
        return False

    if choice.quant.shards > 1:
        console.print(f"[yellow]{choice.quant.label} is split across "
                      f"{choice.quant.shards} files. Download them manually and set "
                      f"llama_model_path to the first shard.[/yellow]")
        return False

    dest = cfg_mod.HOME / "models"
    console.print(f"Downloading {choice.quant.filename} ({choice.quant.size_gb:.1f} GB) "
                  f"to {dest} ...")
    try:
        with console.status("") as status:
            def progress(done, total):
                pct = (done / total * 100) if total else 0
                status.update(f"{done / 1e9:.2f} / {total / 1e9:.2f} GB  ({pct:.0f}%)")
            path = models.download(choice.model.repo, choice.quant.filename, dest, progress)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Download interrupted (resumable — run /models again).[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        return False

    config["llama_model_path"] = path
    config["llama_ctx"] = choice.ctx_k * 1024
    # The KV type is part of the recommendation: it is what makes the context
    # reachable. Leaving the default in place would silently change the maths
    # the menu just showed.
    config["llama_kv_quant"] = choice.kv_quant
    config["model"] = "local"
    cfg_mod.save(config)
    console.print(f"[green]Ready.[/green] {path}\n"
                  f"Context: {choice.ctx_k}K at {config['llama_kv_quant']} KV.")
    return True


# ── Permission ──────────────────────────────────────────────────────────────
# ── Change rendering ────────────────────────────────────────────────────────
# One look for a change, wherever it is shown — the approval prompt, the
# transcript under --accept-all, /diff. Line numbers, a sign, and the changed
# words marked inside an edited line, the way an editor's diff view does it.
# Foreground colour only: a background band that suits a dark terminal is a
# black stripe on a light one.
_KIND_STYLE = {"add": "green", "del": "red", "ctx": "", "gap": "dim"}
_KIND_SIGN = {"add": "+", "del": "-", "ctx": " "}


def _rel(path: str, config: dict) -> str:
    """A path as the user would type it: relative to the working directory."""
    cwd = config.get("_cwd") or os.getcwd()
    try:
        full = os.path.abspath(path if os.path.isabs(path) else os.path.join(cwd, path))
        rel = os.path.relpath(full, cwd)
        return path if rel.startswith("..") else rel
    except ValueError:
        return path


def print_rows(rows: list, limit: int = preview.MAX_DIFF_ROWS, indent: str = "     ") -> None:
    if not rows:
        return
    width = len(str(max((r.new_no or r.old_no or 0) for r in rows)))
    for r in rows[:limit]:
        if r.kind == "gap":
            console.print(f"{indent}{'⋮':>{width}}", style="dim", highlight=False)
            continue
        num = r.old_no if r.kind == "del" else r.new_no
        line = Text(f"{indent}{num:>{width}} ", style="dim")
        body = Text(f"{_KIND_SIGN[r.kind]} {r.text}", style=_KIND_STYLE[r.kind])
        for a, b in r.spans:
            body.stylize("bold reverse", a + 2, b + 2)
        line.append(body)
        console.print(line, highlight=False, soft_wrap=False, overflow="fold")
    if len(rows) > limit:
        console.print(f"{indent}… {len(rows) - limit} more lines — /diff shows "
                      f"everything", style="dim", highlight=False)


def print_change(path: str, before: str | None, after: str, config: dict,
                 pending: bool = False) -> None:
    """A change the way the transcript shows one: a summary line, then the diff."""
    rel = _rel(path, config)
    console.print(f"  ⎿  {preview.summary(rel, before, after, pending)}",
                  style="dim", highlight=False, markup=False)
    if before is None:
        lines = after.splitlines()
        rows = [preview.Row("add", None, i + 1, l.expandtabs(4))
                for i, l in enumerate(lines[:preview.MAX_NEW_FILE_LINES])]
        print_rows(rows)
        if len(lines) > preview.MAX_NEW_FILE_LINES:
            console.print(f"     … {len(lines) - preview.MAX_NEW_FILE_LINES} more "
                          f"lines", style="dim", highlight=False)
    else:
        print_rows(preview.diff_rows(before, after))
    if (loud := preview.gutting(before, after)):
        console.print(f"  {loud}", style="bold red", markup=False, highlight=False)


def choose(question: str, options: list[str], cancel: int) -> int:
    """A small menu: arrows or a number to pick, Enter to confirm, Esc for
    `cancel`. Returns the chosen index.

    The old prompt was the text `[y]es / [n]o / [a]lways` — which Rich read as
    three style tags and swallowed, so it rendered as "es / o / lways" with no
    way to tell what to type.
    """
    import sys as _sys
    if not (_sys.stdin.isatty() and _sys.stdout.isatty()):
        return _choose_by_number(question, options, cancel)
    try:
        import termios, tty, select
    except ImportError:                              # Windows
        return _choose_by_number(question, options, cancel)

    out = _sys.stdout
    current = 0

    def draw(first: bool) -> None:
        if not first:
            out.write(f"\x1b[{len(options)}A")      # back to the first option
        for i, opt in enumerate(options):
            mark = "\x1b[1;36m❯" if i == current else " "
            out.write(f"\r\x1b[2K  {mark} {i + 1}. {opt}\x1b[0m\n")
        out.flush()

    console.print(f"  [bold]{escape(question)}[/bold]  "
                  f"[dim]↑/↓ or 1-{len(options)}, Enter to confirm, Esc for no[/dim]")
    draw(True)
    fd = _sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = os.read(fd, 1)
            if key in (b"\r", b"\n"):
                break
            if key.isdigit() and 1 <= int(key) <= len(options):
                current = int(key) - 1
                draw(False)
                break
            if key in (b"y", b"Y"):
                current = 0
                draw(False)
                break
            if key in (b"n", b"N", b"\x03"):            # n, or Ctrl-C
                current = cancel
                draw(False)
                break
            if key == b"\x1b":
                # A lone Esc, or the start of an arrow key's escape sequence.
                if not select.select([fd], [], [], 0.05)[0]:
                    current = cancel
                    draw(False)
                    break
                seq = os.read(fd, 2)
                if seq in (b"[A", b"OA"):
                    current = (current - 1) % len(options)
                elif seq in (b"[B", b"OB"):
                    current = (current + 1) % len(options)
                draw(False)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    TRANSCRIPT.raw(f"  {question} → {options[current]}\n")
    return current


def _choose_by_number(question: str, options: list[str], cancel: int) -> int:
    console.print(f"  [bold]{escape(question)}[/bold]")
    for i, opt in enumerate(options):
        console.print(f"    {i + 1}. {escape(opt)}")
    try:
        raw = console.input(f"  [dim]1-{len(options)}, Enter for 1:[/dim] ").strip()
    except (EOFError, KeyboardInterrupt):
        return cancel
    if not raw:
        return 0
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return int(raw) - 1
    return {"y": 0, "yes": 0, "a": 1, "always": 1}.get(raw.lower(), cancel)


def make_asker(config: dict, on_approve=None):
    """The permission prompt. `on_approve(name)` runs just before a yes returns,
    so the caller can show progress for the call that is about to run."""
    def ask(name: str, params: dict) -> bool:
        # The tool line itself is already on screen (ToolStart). What is left
        # to say is what the call will actually do. A name and a path is not
        # enough to consent to: the model has replaced a 1,002-line module with
        # three lines, and that write looked exactly like any other from here.
        p = preview.plan(name, params, config)
        if p is not None:
            if p.note:
                for line in p.note.splitlines():
                    console.print(f"     {line}", style="dim", markup=False,
                                  highlight=False)
            if p.after is not None:
                print_change(p.path, p.before, p.after, config, pending=True)
            if p.verdict:
                console.print(f"  {p.verdict}", style="bold red", markup=False,
                              highlight=False)
        what = {"Write": "this write", "Edit": "this edit",
                "Bash": "this command"}.get(name, f"this {name}")
        choice = choose(f"Allow {what}?", [
            "Yes",
            "Yes, and don't ask again this session",
            "No, and tell it what to do instead",
        ], cancel=2)
        if choice == 2:
            try:
                why = console.input("  [dim]What should it do instead? "
                                    "(Enter to just refuse)[/dim] ").strip()
            except (EOFError, KeyboardInterrupt):
                why = ""
            return why or False
        if choice == 1:
            # Underscore-prefixed keys are runtime-only and never saved, so
            # "this session" cannot leak into config.toml the next time
            # anything calls /config.
            config["_accept_session"] = True
            console.print("  [dim]Not asking again until you quit.[/dim]")
        ask.previewed.add(id(params))
        if on_approve:
            on_approve(name, params)
        return True
    ask.previewed = set()
    return ask


# ── Turn rendering ──────────────────────────────────────────────────────────
def _repair_note(state: loop.State) -> str:
    """Repair a resumed transcript before it is ever sent.

    A session only gets repaired on exit if the process caught Ctrl-C. A crash,
    a SIGKILL or a closed laptop leaves a tool_call with no response on disk,
    and resuming it is a guaranteed 400 on the very next request — the failure
    lands one turn later, where it looks like a server problem rather than a
    resume problem.
    """
    n = loop.repair_history(state.messages)
    return f", {n} interrupted tool call(s) repaired" if n else ""


def _tool_detail(name: str, params: dict, config: dict) -> str:
    if (fp := params.get("file_path")):
        return _rel(str(fp), config)
    return str(params.get("command") or params.get("pattern") or params.get("name")
               or params.get("url") or params.get("query") or "")


# What the harness adds to a tool result for the model — hints, "the suite got
# worse", streak notes. They sit at the end of a result, and only a result's
# headline reaches the screen, so the person the last hint is partly *for*
# never saw any of them. Watched on a live run: the monitor could not tell
# whether a hint had fired.
_NOTE_PREFIXES = ("[hint:", "[the suite got worse", "[the suite is green",
                  "[you have read this file", "[this is ")


def _show_notes(result: str) -> None:
    for line in (result or "").splitlines()[-6:]:
        line = line.strip()
        if line.startswith(_NOTE_PREFIXES):
            console.print(f"     {line}", style="yellow", markup=False, highlight=False)


def _result_line(name: str, result: str) -> tuple[str, int]:
    """The one line of a result worth showing, and how many lines there were.

    For a command that is the *last* line: pytest, make, cargo and most other
    tools print progress first and the verdict at the end, so the first line of
    a passing test run was a row of dots.
    """
    import re as _re
    lines = [l.strip() for l in (result or "").splitlines() if l.strip()]
    if not lines:
        return "", 0
    if name == "Read" and not lines[0].startswith("Error"):
        return f"Read {len(lines)} line{'s' * (len(lines) != 1)}", 1   # the content is for the model
    if name != "Bash":
        return lines[0], len(lines)
    tail = lines[-1]
    if _re.fullmatch(r"\[exit \d+\]", tail) and len(lines) > 1:
        tail = f"{lines[-2]}  {tail}"
    return _re.sub(r"\s{2,}", "  ", tail), len(lines)


def _context_note(state: loop.State, config: dict) -> str:
    size = int(config.get("llama_ctx") or 0)          # display only, not a budget
    if not size:
        return ""
    used = context.estimate_tokens(state.messages, state.system)
    return f"context {100 * used / size:.0f}% of {size / 1024:.0f}k"


def run_turn(state: loop.State, config: dict, tracker) -> None:
    """Run one user message to completion, rendering events as they arrive."""
    status = None                       # a live spinner, while something runs

    def stop_status():
        nonlocal status
        if status is not None:
            status.stop()
            status = None

    def start_status(text: str):
        nonlocal status
        stop_status()
        status = console.status(f"[dim]{text}[/dim]", spinner="dots")
        status.start()

    def end_thinking():
        """Close a thinking block: one line of record where the counter was."""
        nonlocal thinking
        if not thinking:
            return
        thinking = False
        stop_status()
        if show_think:
            console.print()
            return
        # To the last reasoning chunk, not to whatever event ended the block:
        # a tool call streams its arguments silently, and that time is not
        # thinking.
        spent = think_last - think_start
        TRANSCRIPT.raw("\n── end of thinking ──\n")
        console.print(f"✻ thought for {spent:.0f}s · ~{think_bytes // 4:,} tokens",
                      style="dim", highlight=False, markup=False)

    def on_approve(name, params):
        if name == "Bash":
            start_status(f"running {str(params.get('command', ''))[:60]} … "
                         f"Ctrl-C interrupts")

    asker = make_asker(config, on_approve)
    buffer: list[str] = []
    think_buf: list[str] = []
    streaming = False
    thinking = False
    think_start = last_tick = think_last = 0.0
    think_bytes = 0
    show_think = bool(config.get("show_thinking"))
    turn_start = _time.monotonic()
    calls = rounds = 0
    changed: dict[str, None] = {}           # ordered set of files written
    pending: dict[int, tuple] = {}          # id(params) -> (path, before, t0)

    try:
        for event in loop.run(state, config, asker, tracker):
            kind = type(event).__name__
            if kind != "ThinkChunk":
                end_thinking()
                if kind != "ToolDraft":
                    stop_status()
            if kind == "ToolDraft":
                # A tool call's content streaming in: a file being written, a
                # long command. Nothing else arrives while it does.
                what = {"Write": "writing", "Edit": "editing"}.get(event.name, "preparing")
                line = f"{what} {event.name or 'a tool call'} … {event.chars:,} chars"
                if status is None:
                    start_status(line)
                else:
                    status.update(f"[dim]{line}[/dim]")
                continue
            if kind == "TextChunk":
                streaming = True
                buffer.append(event.text)
                console.print(event.text, end="", markup=False, highlight=False)
            elif kind == "ThinkChunk":
                # Never stored in history (see provider.ThinkFilter). Collapsed
                # by default: reasoning is usually noise, but you want to be
                # able to look when the model does something baffling.
                think_buf.append(event.text)
                if not thinking:
                    if streaming:
                        console.print()
                    thinking, streaming = True, False
                    think_start = _time.monotonic()
                    think_bytes = 0
                    last_tick = 0.0
                    if show_think:
                        console.print("[dim]thinking…[/dim]", end="")
                think_bytes += len(event.text)
                think_last = _time.monotonic()
                if show_think:
                    console.print(event.text, end="", style="dim",
                                  markup=False, highlight=False)
                else:
                    # Hidden on screen, kept in the log: the text is what tells
                    # a model working hard from one going round in circles.
                    if think_bytes == len(event.text):
                        TRANSCRIPT.raw("── thinking ──\n")
                    TRANSCRIPT.raw(event.text)
                    # A static "thinking…" is indistinguishable from a hang, and
                    # was twice diagnosed as one, so it counts up. It runs on
                    # the live status line: this used to print "\r" + text,
                    # but Rich strips control characters from printed text, so
                    # every update was *appended* — and past the terminal's
                    # width, wrapped onto a new line twice a second.
                    now = _time.monotonic()
                    if now - last_tick >= 0.5:
                        last_tick = now
                        line = (f"thinking… ~{think_bytes // 4:,} tokens, "
                                f"{now - think_start:.0f}s")
                        if status is None:
                            start_status(line)
                        else:
                            status.update(f"[dim]{line}[/dim]")
            elif kind == "ToolStart":
                if streaming:
                    console.print()
                    streaming = False
                calls += 1
                rounds = max(rounds, event.round)
                console.print(Text.assemble(
                    ("● ", "cyan"), (event.name, "bold"),
                    (f"({_tool_detail(event.name, event.params, config)[:100]})", "")),
                    highlight=False)
                before = None
                if event.name in ("Write", "Edit") and event.params.get("file_path"):
                    target = tools._resolve(str(event.params["file_path"]), config)
                    before = preview._read(target) if target.exists() else None
                    pending[id(event.params)] = (str(event.params["file_path"]),
                                                 before, _time.monotonic())
                else:
                    pending[id(event.params)] = ("", None, _time.monotonic())
                    # Nothing is asked for a read-only call, and nothing at all
                    # under --accept-all, so the spinner starts here; otherwise
                    # it starts once the call has been approved.
                    if (event.name not in tools.MUTATING or config.get("accept_all")
                            or config.get("_accept_session")):
                        start_status(f"running {event.name} … Ctrl-C interrupts")
            elif kind == "ToolEnd":
                failed = event.denied or event.result.startswith("Error")
                # Match the end to its start by the call's params object, which
                # the loop passes through unchanged.
                key = next(reversed(pending), None)
                path, before, t0 = pending.pop(key, ("", None, _time.monotonic()))
                spent = _time.monotonic() - t0
                took = f" · {spent:.1f}s" if spent >= 1.0 else ""
                if event.name in ("Write", "Edit") and path and not failed:
                    target = tools._resolve(path, config)
                    after = preview._read(target) if target.exists() else ""
                    changed[_rel(path, config)] = None
                    # Shown already, in the approval prompt: one line will do.
                    shown = key in asker.previewed
                    asker.previewed.discard(key)
                    if shown:
                        console.print(f"  ⎿  {preview.summary(_rel(path, config), before, after)}{took}",
                                      style="dim", markup=False, highlight=False)
                    else:
                        print_change(path, before, after, config)
                    continue
                head, n = _result_line(event.name, event.result)
                more = f"  (+{n - 1} lines)" if n > 1 else ""
                # style= not markup: tool output contains brackets that Rich
                # would otherwise try to parse as tags.
                console.print(f"  ⎿  {head[:140]}{more}{took}",
                              style="red" if failed else "dim",
                              markup=False, highlight=False)
                _show_notes(event.result)
            elif kind == "Compacting":
                if streaming:
                    console.print()
                    streaming = False
                # A model call: tens of seconds that would otherwise be silence.
                start_status(f"compacting ~{event.tokens:,} tokens of history — "
                             f"the model is writing a note of what matters")
            elif kind == "Notice":
                if streaming:
                    console.print()
                    streaming = False
                console.print(f"[yellow]{escape(event.text)}[/yellow]", highlight=False)
            elif kind == "TurnDone":
                if streaming:
                    console.print()
                    streaming = False
    except KeyboardInterrupt:
        end_thinking()
        stop_status()
        console.print("\n[yellow]interrupted[/yellow]")
        # Keep history well-formed: a dangling assistant tool_call with no tool
        # response is a guaranteed 400 on the next request.
        if (n := loop.repair_history(state.messages)):
            console.print(f"[dim]  ({n} pending tool call(s) marked interrupted — "
                          f"everything before them is kept)[/dim]")
    except Exception as e:
        stop_status()
        console.print(f"[red]{type(e).__name__}: {escape(str(e))}[/red]")
        return
    finally:
        end_thinking()
        stop_status()

    state.last_thinking = "".join(think_buf)
    if state.last_thinking and not show_think:
        n = len(state.last_thinking) // 4
        console.print(f"[dim]  (~{n:,} tokens of reasoning — /think to view)[/dim]",
                      highlight=False)

    session.append(state.session_id, {"messages": state.messages})

    # Compact if we're near the window. Never touches the head of the prompt.
    # Same budget the loop uses, so a turn cannot start already over it.
    budget = cfg_mod.history_budget(config, tools.schemas_for(config))
    used = context.estimate_tokens(state.messages, state.system)
    if used > budget and config.get("model"):
        t0 = _time.monotonic()
        with console.status(f"[dim]compacting ~{used:,} tokens of history — the "
                            f"model is writing a note of what matters[/dim]"):
            state.messages, freed = context.compact_with_model(
                state.messages, int(budget * float(config.get('compact_to', 0.6))),
                config['model'], state.system, config)
        if freed:
            console.print(f"[dim]compacted: reclaimed ~{freed:,} tokens in "
                          f"{_time.monotonic() - t0:.0f}s[/dim]", highlight=False)

    # The footer: what the turn cost and what it changed. The last part is the
    # one that matters — the undo exists, and nothing used to say so.
    parts = [f"{_time.monotonic() - turn_start:.0f}s"]
    if calls:
        parts.append(f"{calls} tool call{'s' * (calls != 1)}")
        if (mx := int(config.get("max_turns", 100))) and rounds >= mx * 0.8:
            parts.append(f"{rounds} of {mx} rounds")
    if changed:
        parts.append(f"{len(changed)} file{'s' * (len(changed) != 1)} changed")
    if (ctx := _context_note(state, config)):
        parts.append(ctx)
    console.print(f"[dim]✓ {' · '.join(parts)}[/dim]", highlight=False)
    if changed:
        console.print("[dim]  /diff to review · /rewind to undo[/dim]", highlight=False)


def show_unified(stat: str, text: str, title: str) -> None:
    """A whole unified diff, coloured, through the pager when it is long."""
    def emit():
        console.print(f"[bold]{escape(title)}[/bold]")
        for line in stat.splitlines():
            console.print("  " + line, style="dim", markup=False, highlight=False)
        console.print()
        for line in text.splitlines():
            style = ("bold" if line.startswith(("diff --git", "+++", "---"))
                     else "cyan" if line.startswith("@@")
                     else "green" if line.startswith("+")
                     else "red" if line.startswith("-") else "")
            console.print(line, style=style, markup=False, highlight=False,
                          soft_wrap=True)
    if len(text.splitlines()) > (console.size.height or 40) - 4:
        with console.pager(styles=True):
            emit()
    else:
        emit()


# ── Things typed into a message ─────────────────────────────────────────────
# Output of `!command`, waiting to go out with the next message. It is not sent
# on its own: a user message with no question in it would make the model guess
# what to do with it, and two user messages in a row break the alternation
# some chat templates require.
_PENDING_SHELL: list[str] = []


def run_shell(cmd: str, config: dict) -> None:
    """The user's own command. Not jailed: the jail exists to bound the
    *model*, and the person at the keyboard already has a shell."""
    import subprocess
    try:
        r = subprocess.run(cmd, shell=True, cwd=config["_cwd"], capture_output=True,
                           text=True, timeout=600)
        out = (r.stdout + r.stderr).rstrip()
        if r.returncode:
            out += f"\n[exit {r.returncode}]"
    except subprocess.TimeoutExpired:
        out = "[timed out after 600s]"
    for line in (out or "[no output]").splitlines():
        console.print(line, style="dim", markup=False, highlight=False)
    _PENDING_SHELL.append(f"I ran `{cmd}` myself:\n```\n"
                          f"{tools._truncate(out or '[no output]', tools.output_cap(config))}"
                          f"\n```")
    console.print("[dim]  (goes with your next message)[/dim]")


def compose_message(line: str, config: dict, tracker) -> str:
    """The message as sent: shell output first, then the text, then any @files.

    An @file is attached with the same Read the model would have issued —
    same jail, same cap, and marked as read — so the model can edit it on
    its first call instead of spending one turn discovering it.
    """
    import re as _re
    parts = list(_PENDING_SHELL)
    _PENDING_SHELL.clear()
    parts.append(line)
    seen = set()
    for m in _re.finditer(r"(?<!\S)@([^\s]+)", line):
        ref = m.group(1).rstrip(".,;:!?)")
        target = tools._resolve(ref, config)
        if ref in seen or not target.is_file():
            continue
        seen.add(ref)
        body = tools.dispatch("Read", {"file_path": ref}, config, tracker)
        if body.startswith("Error"):
            console.print(f"[yellow]not attached: {escape(body)}[/yellow]")
            continue
        parts.append(f"Contents of `{ref}`, attached by me:\n{body}")
        console.print(f"[dim]  attached {escape(ref)}[/dim]")
    return "\n\n".join(parts)


def _commands() -> list[str]:
    import re as _re
    return sorted(set(_re.findall(r"^\s+(/[a-z]+)", HELP, _re.M)) | {"/exit"})


def make_completer(config: dict):
    """Tab completion for /commands and @paths. None without prompt_toolkit."""
    try:
        from prompt_toolkit.completion import Completer, Completion
    except ImportError:
        return None
    commands = _commands()
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache",
            ".pytest_cache", ".ruff_cache"}

    class _C(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if text.startswith("/") and " " not in text:
                for c in commands:
                    if c.startswith(text):
                        yield Completion(c, -len(text))
                return
            word = document.get_word_before_cursor(WORD=True)
            if not word.startswith("@"):
                return
            frag = word[1:]
            head, _, prefix = frag.rpartition("/")
            base = os.path.join(config["_cwd"], head)
            try:
                names = sorted(os.listdir(base))
            except OSError:
                return
            for name in names:
                if name in skip or (name.startswith(".") and not prefix.startswith(".")):
                    continue
                if name.startswith(prefix):
                    full = os.path.join(base, name)
                    rel = f"{head}/{name}" if head else name
                    tail = "/" if os.path.isdir(full) else ""
                    yield Completion("@" + rel + tail, -len(word), display=name + tail)
    return _C()


# ── Slash commands ──────────────────────────────────────────────────────────
def handle_command(line: str, state: loop.State, config: dict, tracker) -> bool:
    """Returns False to exit the REPL."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return False

    if cmd == "/help":
        console.print(HELP)

    elif cmd == "/model":
        if not arg:
            console.print(f"model: [bold]{config['model'] or '(none)'}[/bold]")
        else:
            config["model"] = arg
            cfg_mod.save(config)
            console.print(f"model -> [bold]{arg}[/bold]")

    elif cmd == "/models":
        if choose_model(config):
            state.system = context.build_system(config)

    elif cmd == "/config":
        if not arg:
            for k in sorted(cfg_mod.DEFAULTS):
                mark = " *" if config.get(k) != cfg_mod.DEFAULTS[k] else ""
                console.print(f"  {k:<24} {config.get(k)!r}{mark}")
        elif "=" in arg:
            key, _, val = arg.partition("=")
            try:
                new = cfg_mod.set_value(config, key.strip(), val.strip())
                console.print(f"{key.strip()} = {new!r}")
            except KeyError as e:
                console.print(f"[red]{e}[/red]")
            else:
                # Changing the window on a running server silently invalidates
                # every budget derived from it — and this is advice the harness
                # itself gives when a start fails for want of VRAM. Restart so
                # the number means something.
                from .provider import split_model as _split
                if key.strip() == "llama_ctx" and _split(config["model"])[0] == "local":
                    if (running := server.live_ctx(config)) and running != int(new):
                        console.print(f"[dim]server is serving {running} tokens; "
                                      f"restarting it at {new}…[/dim]")
                        server.stop()
                        if server.ensure(config):
                            console.print("[dim]server restarted[/dim]")
                        else:
                            console.print("[red]restart failed — the old context "
                                          "is gone and the new one did not start. "
                                          "Run /models or set llama_ctx lower.[/red]")
        else:
            console.print(f"  {arg} = {config.get(arg)!r}")

    elif cmd == "/compact":
        # Asked for explicitly, so aim below the automatic trigger — otherwise
        # /compact right after an automatic pass would do nothing at all.
        before = context.estimate_tokens(state.messages, state.system)
        target = cfg_mod.history_budget(config, tools.schemas_for(config)) // 2
        state.messages, freed = context.compact_with_model(
            state.messages, target, config['model'], state.system, config)
        console.print(f"[dim]{before} -> {before - freed} tokens[/dim]")

    elif cmd == "/think":
        if arg in ("on", "off"):
            cfg_mod.set_value(config, "show_thinking", arg)
            console.print(f"live reasoning display: [bold]{arg}[/bold]")
        elif arg.startswith("save"):
            # A runaway's reasoning is the one thing needed to tell a model
            # that is working hard from one that is going round in circles,
            # and the screen is the wrong place to study 10,000 tokens of it.
            if not state.last_thinking:
                console.print("No reasoning captured from the last turn.")
                return True
            target = (pathlib.Path(arg[4:].strip()).expanduser() if arg[4:].strip()
                      else cfg_mod.HOME / "last_thinking.md")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(state.last_thinking, encoding="utf-8")
            console.print(f"Saved ~{len(state.last_thinking) // 4:,} tokens of "
                          f"reasoning to [bold]{escape(str(target))}[/bold]")
        elif state.last_thinking:
            # markup=False: reasoning is full of brackets, and Rich would read
            # "[a]" in it as a style tag and silently drop it.
            console.print(state.last_thinking, style="dim", markup=False,
                          highlight=False)
        else:
            console.print("No reasoning captured from the last turn. "
                          "(/think on streams it live.)")

    elif cmd == "/undo":
        console.print(tools.undo_last())

    elif cmd == "/diff":
        rows = checkpoint.history(config, state.session_id)
        if not rows:
            console.print("Nothing has changed this session."
                          if checkpoint.enabled(config)
                          else "Checkpoints are off (or git is not installed), so "
                               "there is no record of what changed.")
            return True
        ref = arg.lstrip("-")
        if ref.isdigit():
            if int(ref) >= len(rows):
                console.print(f"Only {len(rows)} checkpoint(s); the oldest is -{len(rows) - 1}.")
                return True
            target, what = rows[int(ref)][0], f"the last {ref} change(s)"
        else:
            target, what = checkpoint.baseline(config, state.session_id), "this session"
        text = checkpoint.diff(config, state.session_id, target)
        if not text.strip():
            console.print(f"No changes in {what}.")
            return True
        show_unified(checkpoint.changes_since(config, state.session_id, target), text,
                     f"Changes in {what}")

    elif cmd == "/clear":
        # A new conversation, not a new working tree: the files stay as they
        # are, and the previous conversation is still on disk for /resume.
        old_id = state.session_id
        fresh = loop.State(system=state.system, session_id=session.new_id(config["_cwd"]))
        for f in ("messages", "continuations", "empty_retries", "last_thinking",
                  "recent_thinking", "ledger", "compact_floor", "session_id"):
            setattr(state, f, getattr(fresh, f))
        if tracker is not None:
            tracker._read.clear()
        _PENDING_SHELL.clear()
        if TRANSCRIPT.path:
            TRANSCRIPT.open(state.session_id)
        console.print(f"Started a new conversation. The last one is kept: "
                      f"[bold]/resume {old_id}[/bold]")

    elif cmd == "/log":
        if TRANSCRIPT.path:
            console.print(f"This session's log: [bold]{escape(str(TRANSCRIPT.path))}[/bold]\n"
                          f"[dim]tail -f it to follow a run from another terminal.[/dim]")
        else:
            console.print("No log for this session (/config transcript=true, then restart).")

    elif cmd == "/checkpoints":
        rows = checkpoint.history(config, state.session_id)
        if not rows:
            console.print("No checkpoints yet."
                          if checkpoint.enabled(config)
                          else "Checkpoints are off (or git is not installed).")
            return True
        for i, (sha, when, label) in enumerate(rows):
            marker = "now" if i == 0 else f" -{i}"
            console.print(f"  [bold]{marker:>4}[/bold] [dim]{sha}  {when:<16}[/dim] "
                          f"{escape(label)}")
        console.print("[dim]/rewind -2  restores the tree to that point[/dim]")

    elif cmd == "/rewind":
        rows = checkpoint.history(config, state.session_id)
        if not rows:
            console.print("Nothing to rewind to.")
            return True
        # `-2` means two changes ago, which is how the list above reads. A bare
        # sha works too; a bare /rewind means the last checkpoint.
        ref = arg.lstrip("-") or "1"
        if ref.isdigit():
            if int(ref) >= len(rows):
                console.print(f"Only {len(rows)} checkpoint(s); "
                              f"the oldest is -{len(rows) - 1}.")
                return True
            target = rows[int(ref)][0]
        elif arg:
            target = arg
        else:
            console.print("Nothing has changed yet.")
            return True
        stat = checkpoint.changes_since(config, state.session_id, target)
        if not stat:
            console.print(f"Already at {target}; nothing would change.")
            return True
        console.print(f"[yellow]Rewinding to {target} would undo:[/yellow]")
        for line in stat.splitlines():
            console.print("  " + line, markup=False, style="dim", highlight=False)
        console.print("[dim]Files created since then are deleted. Your own git "
                      "history is not touched.[/dim]")
        try:
            if not console.input("  rewind? [y/N]: ").strip().lower().startswith("y"):
                console.print("Left alone.")
                return True
        except (EOFError, KeyboardInterrupt):
            console.print("Left alone.")
            return True
        console.print(checkpoint.restore(config, state.session_id, target))

    elif cmd == "/resume":
        sid = arg or session.latest()
        if not sid:
            console.print("No sessions found.")
            return True
        records = session.load(sid)
        if not records:
            console.print(f"[red]Nothing in session {sid}[/red]")
            return True
        state.messages = records[-1].get("messages", [])
        state.session_id = sid
        if TRANSCRIPT.path:
            TRANSCRIPT.open(sid)
        if tracker is not None:
            fresh = context.FileTracker.from_messages(state.messages,
                                                      config.get("_cwd"))
            tracker._read = fresh._read
        note = _repair_note(state)
        console.print(f"Resumed [bold]{sid}[/bold] ({len(state.messages)} messages){note}")

    elif cmd == "/research":
        if not arg:
            for p in research.list_workspaces()[:10]:
                console.print(f"  {p.stem}  [dim]{p}[/dim]")
            return True
        console.print(f"[dim]researching: {arg}[/dim]")

        def on_event(kind, text):
            if kind == "tool":
                console.print(f"[dim]-> {text}[/dim]")
        try:
            path = research.run(arg, config, on_event)
            console.print(f"[green]Report:[/green] {path}")
        except KeyboardInterrupt:
            console.print("\n[yellow]stopped (resumable)[/yellow]")
        except Exception as e:
            console.print(f"[red]{type(e).__name__}: {e}[/red]")

    else:
        console.print(f"[red]Unknown command {cmd}[/red] — try /help")

    return True


# ── Entry point ─────────────────────────────────────────────────────────────
def _install_exit_handlers() -> None:
    """Stop the server and close the log however the harness ends.

    SIGTERM (a timeout, `kill`) and SIGHUP (closing the terminal) used to end
    the process without running any cleanup, and one-shot `-p` mode returned
    before the cleanup line even on success — each leaving llama-server
    running with the GPU. Turning both signals into an ordinary exit lets
    atexit do the same work a /quit does.
    """
    import atexit
    import signal
    atexit.register(server.stop)
    atexit.register(TRANSCRIPT.close)
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            try:
                signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))
            except (ValueError, OSError):
                pass          # not the main thread, or not supported here


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="miniharness",
                                 description="A minimal local-first coding agent")
    ap.add_argument("-p", "--prompt", help="one-shot: run this and exit")
    ap.add_argument("-m", "--model", help="model to use (e.g. gpt-4o, ollama/qwen3, local)")
    ap.add_argument("-C", "--cwd", default=os.getcwd(), help="working directory")
    ap.add_argument("--accept-all", action="store_true", help="never ask permission")
    ap.add_argument("--resume", nargs="?", const="", help="resume a session")
    ap.add_argument("--no-repo-map", action="store_true", help="omit the repo map")
    args = ap.parse_args(argv)
    _install_exit_handlers()

    config = cfg_mod.load()
    config["_cwd"] = os.path.abspath(args.cwd)
    if args.model:
        config["model"] = args.model
    if args.accept_all:
        config["accept_all"] = True
    if args.no_repo_map:
        config["repo_map"] = False

    # First run: nothing configured at all.
    if not config.get("model"):
        console.print("[bold]MiniHarness[/bold] — no model configured.\n")
        if not choose_model(config):
            console.print("Point at any OpenAI-compatible endpoint instead, e.g.:\n"
                          "  miniharness -m ollama/qwen3-coder\n"
                          "  miniharness -m gpt-4o        (needs OPENAI_API_KEY)")
            return 1

    # Local model: make sure a server is up.
    from .provider import split_model
    if split_model(config["model"])[0] == "local":
        try:
            if not server.ensure(config):
                console.print("[yellow]No llama-server reachable and autostart is "
                              "off or unconfigured.[/yellow]")
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            return 1

    tracker = context.FileTracker()
    state = loop.State(system=context.build_system(config))
    state.session_id = session.new_id(config["_cwd"])

    if args.resume is not None:
        sid = args.resume or session.latest()
        if sid and (records := session.load(sid)):
            state.messages = records[-1].get("messages", [])
            state.session_id = sid
            # Restore what the session had already read; otherwise the first
            # Edit after a resume is refused for a file already in context.
            tracker = context.FileTracker.from_messages(state.messages,
                                                        config.get("_cwd"))
            note = _repair_note(state)
            console.print(f"[dim]resumed {sid} ({len(state.messages)} messages){note}[/dim]")

    if config.get("transcript", True):
        TRANSCRIPT.open(state.session_id)
        TRANSCRIPT.raw(f"model {config['model']} · {config['_cwd']}\n")

    # Get the stable prefix into the KV cache before the first real turn.
    if (status := server.warm_prefix(config, state.system, tools.schemas_for(config))):
        console.print(f"[dim]{status}[/dim]")

    # Verify the server actually reuses that prefix. A forked build can pin
    # every turn at full prefill without erroring — silent, and enormous.
    if config.get("check_prefix_cache", True) and split_model(config["model"])[0] == "local":
        ok, msg = server.check_prefix_cache(config)
        if ok is False:
            console.print(f"[yellow]warning: {msg}[/yellow]")
        elif ok:
            console.print(f"[dim]{msg}[/dim]")

    if args.prompt:
        TRANSCRIPT.raw(f"\n[{_time.strftime('%H:%M:%S')}] > {args.prompt}\n")
        state.add_user(compose_message(args.prompt, config, tracker))
        run_turn(state, config, tracker)
        return 0

    console.print(f"[bold]MiniHarness[/bold] [dim]{config['model']} · "
                  f"{config['_cwd']}[/dim]  —  /help")
    if TRANSCRIPT.path:
        console.print(f"[dim]log: {escape(str(TRANSCRIPT.path))}[/dim]", highlight=False)

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        cfg_mod.HOME.mkdir(parents=True, exist_ok=True)
        psession = PromptSession(history=FileHistory(str(cfg_mod.HOME / "history")),
                                 completer=make_completer(config),
                                 complete_while_typing=False)
        read_line = lambda: psession.prompt("> ")  # noqa: E731
    except ImportError:
        read_line = lambda: input("> ")  # noqa: E731

    while True:
        try:
            line = read_line().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        # The prompt is drawn by prompt_toolkit, not by the console, so what
        # was typed has to be recorded by hand.
        TRANSCRIPT.raw(f"\n[{_time.strftime('%H:%M:%S')}] > {line}\n")
        if line.startswith("/"):
            if not handle_command(line, state, config, tracker):
                break
            continue
        if line.startswith("!"):
            if (cmd := line[1:].strip()):
                run_shell(cmd, config)
            continue
        state.add_user(compose_message(line, config, tracker))
        run_turn(state, config, tracker)

    TRANSCRIPT.close()
    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
