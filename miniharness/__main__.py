"""CLI and REPL.

Seven slash commands, one file. Promethean had eleven command modules totalling
5,848 lines; almost all of it was surface for features that no longer exist.
"""

from __future__ import annotations

import argparse
import os
import sys
import time as _time

from rich.console import Console
from rich.markup import escape

from . import config as cfg_mod
from . import checkpoint, context, loop, models, preview, research, server, session, tools

console = Console()

HELP = """\
[bold]Commands[/bold]
  /help                 this list
  /model [name]         show or switch model (e.g. gpt-4o, ollama/qwen3, local)
  /models               pick and download a local model for this machine
  /config [k=v]         show or set configuration
  /compact              shrink the conversation now
  /think [on|off]       show the last turn's reasoning, or toggle live display
  /undo                 revert the last file write or edit
  /checkpoints          list the snapshots taken after each accepted change
  /rewind [-n|sha]      restore the working tree to a checkpoint
  /resume [id]          resume a previous session (default: the most recent)
  /research <question>  spawn a bounded research run; writes a markdown report
  /quit                 exit

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
def _line_style(line: str) -> str:
    if line.startswith(("!!", "will be REFUSED", "will FAIL")):
        return "bold red"
    if line.startswith("+"):
        return "green"
    if line.startswith("-"):
        return "red"
    if line.startswith("@@"):
        return "cyan"
    return "dim"


def make_asker(config: dict):
    def ask(name: str, params: dict) -> bool:
        if name == "Bash":
            detail = params.get("command", "")
        else:
            detail = params.get("file_path", "")
        console.print(f"\n[bold yellow]{name}[/bold yellow] {escape(str(detail))}")
        # Say what the call will actually do. A name and a path is not enough
        # to consent to: the model has replaced a 1,002-line module with three
        # lines, and that write looked exactly like any other from here.
        # markup=False because file content is not rich markup — a line
        # containing [dim] is a line of code.
        for line in preview.describe(name, params, config).splitlines():
            console.print("  " + line, markup=False, style=_line_style(line),
                          highlight=False)
        try:
            answer = console.input("  [y]es / [n]o / [a]lways: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if answer.startswith("a"):
            config["accept_all"] = True
            return True
        return answer.startswith("y") or answer == ""
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


def run_turn(state: loop.State, config: dict, tracker) -> None:
    """Run one user message to completion, rendering events as they arrive."""
    asker = make_asker(config)
    buffer: list[str] = []
    think_buf: list[str] = []
    streaming = False
    thinking = False
    think_start = last_tick = 0.0
    think_bytes = 0
    show_think = bool(config.get("show_thinking"))

    try:
        for event in loop.run(state, config, asker, tracker):
            kind = type(event).__name__
            if kind == "TextChunk":
                if thinking:
                    console.print()
                    thinking = False
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
                if show_think:
                    console.print(event.text, end="", style="dim",
                                  markup=False, highlight=False)
                else:
                    # A static "thinking…" is indistinguishable from a hang, and
                    # was twice diagnosed as one. It used to be bounded by the
                    # continuation notices, which fired every time the reply cap
                    # was hit — but the cap is gone (§2.8), so a turn can now
                    # deliberate for minutes with nothing between "thinking…"
                    # and the tool call. Counting the chunks already arriving
                    # costs nothing and needs no server-specific endpoint.
                    now = _time.monotonic()
                    if now - last_tick >= 0.5:
                        last_tick = now
                        console.print(
                            f"\r[dim]thinking… ~{think_bytes // 4:,} tokens, "
                            f"{now - think_start:.0f}s[/dim]  ",
                            end="")
            elif kind == "ToolStart":
                if streaming or thinking:
                    console.print()
                    streaming = thinking = False
                detail = (event.params.get("command")
                          or event.params.get("file_path")
                          or event.params.get("pattern")
                          or event.params.get("name")
                          or event.params.get("url") or "")
                console.print(f"[dim]-> {event.name} {str(detail)[:100]}[/dim]")
            elif kind == "ToolEnd":
                colour = "red" if event.denied or event.result.startswith("Error") else "dim"
                lines = event.result.splitlines() if event.result else [""]
                more = f"  (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
                # style= not markup: tool output contains brackets that Rich
                # would otherwise try to parse as tags.
                console.print(f"   {lines[0][:140]}{more}",
                              style=colour, markup=False, highlight=False)
            elif kind == "Notice":
                console.print(f"[yellow]{event.text}[/yellow]")
            elif kind == "TurnDone":
                if streaming or thinking:
                    console.print()
                    streaming = thinking = False
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        # Keep history well-formed: a dangling assistant tool_call with no tool
        # response is a guaranteed 400 on the next request.
        if (n := loop.repair_history(state.messages)):
            console.print(f"[dim]  ({n} pending tool call(s) marked interrupted)[/dim]")
    except Exception as e:
        console.print(f"[red]{type(e).__name__}: {e}[/red]")
        return

    state.last_thinking = "".join(think_buf)
    if state.last_thinking and not show_think:
        n = len(state.last_thinking) // 4
        console.print(f"[dim]  (~{n} tokens of reasoning — /think to view)[/dim]")

    session.append(state.session_id, {"messages": state.messages})

    # Compact if we're near the window. Never touches the head of the prompt.
    # Same budget the loop uses, so a turn cannot start already over it.
    budget = cfg_mod.history_budget(config, tools.schemas_for(config))
    used = context.estimate_tokens(state.messages, state.system)
    if used > budget:
        state.messages, freed = context.compact_with_model(
            state.messages, int(budget * float(config.get('compact_to', 0.6))),
            config['model'], state.system, config)
        if freed:
            console.print(f"[dim]compacted: reclaimed ~{freed} tokens[/dim]")


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
        elif state.last_thinking:
            console.print(f"[dim]{state.last_thinking}[/dim]")
        else:
            console.print("No reasoning captured from the last turn. "
                          "(/think on streams it live.)")

    elif cmd == "/undo":
        console.print(tools.undo_last())

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
        state.add_user(args.prompt)
        run_turn(state, config, tracker)
        return 0

    console.print(f"[bold]MiniHarness[/bold] [dim]{config['model']} · "
                  f"{config['_cwd']}[/dim]  —  /help")

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        cfg_mod.HOME.mkdir(parents=True, exist_ok=True)
        psession = PromptSession(history=FileHistory(str(cfg_mod.HOME / "history")))
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
        if line.startswith("/"):
            if not handle_command(line, state, config, tracker):
                break
            continue
        state.add_user(line)
        run_turn(state, config, tracker)

    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
