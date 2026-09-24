"""Configuration: a flat dict, a TOML file, and defaults.

No schema layer, no validation framework, no nested namespaces. Config is a flat
``dict[str, Any]`` persisted to ``~/.miniharness/config.toml``. Adding a setting
means adding one line to DEFAULTS.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10
    tomllib = None  # type: ignore[assignment]


HOME = Path(os.environ.get("MINIHARNESS_HOME", Path.home() / ".miniharness"))
CONFIG_PATH = HOME / "config.toml"

#: Assumed window when the server has not told us and nothing is configured.
DEFAULT_CONTEXT = 32_768


def window(cfg: dict) -> int:
    """The context length every other budget is derived from."""
    return int(cfg.get("llama_ctx") or 0) or DEFAULT_CONTEXT


def budget(cfg: dict, share_key: str, floor: int = 256) -> int:
    """Tokens allowed for ``share_key``, as a fraction of the window."""
    return max(floor, int(window(cfg) * float(cfg.get(share_key, 0.25))))


def history_budget(cfg: dict, schemas: list | None = None) -> int:
    """Tokens the conversation may occupy.

    The history is not the whole request. Every request also carries the tool
    schemas, the per-turn focus map, and a reservation for the reply the server
    is about to generate — none of which compaction can shrink. Budgeting the
    messages against the raw window overflows by exactly the size of everything
    else, which the server returns as a 400 that kills the run mid-turn:

        request (17056 tokens) exceeds the available context size (16384)

    So ``compact_at`` is a fraction of the room actually left for history, not
    of the whole window. Lives here, beside the other shares, because three
    separate call sites computed it independently and two got it wrong.
    """
    import json

    # Nothing is held back for the reply.
    #
    # A reservation limits in both directions: it caps the answer at the size
    # reserved, and it shrinks the history by that much whether or not the
    # answer needs it. The reply is the work; the history is what should yield.
    # So history is allowed to grow to `compact_at` of what is genuinely free,
    # the reply takes whatever room it needs, and if it runs out, compaction
    # frees window and the reply resumes (`max_truncation_resumes`) — after the
    # fact, on demand, rather than by permanent set-aside.
    #
    # Schemas and the repo map are still subtracted because they are sent on
    # every request and are not negotiable at generation time.
    overhead = (
        (len(json.dumps(schemas)) // 4 if schemas else 0)
        + budget(cfg, "repo_map_share", floor=0)
    )
    room = max(1024, window(cfg) - overhead)
    return max(512, int(room * float(cfg.get("compact_at", 0.85))))


DEFAULTS: dict[str, Any] = {
    # ── Model / provider ────────────────────────────────────────────────────
    "model": "",                     # e.g. "local", "gpt-4o", "ollama/qwen3-coder"
    "base_url": "",                  # explicit override; else derived from provider
    "api_key_env": "",               # explicit override; else derived from provider
    # Sampling. Empty means "the model's own": llama-server is launched with the
    # model card's values (models.FAMILY_SAMPLING), and a request that sends
    # nothing gets them. Set one here only to overrule the model's authors.
    "temperature": "",
    "top_p": "",
    "top_k": "",
    "min_p": "",
    "presence_penalty": "",
    # An *idle* bound: how long to wait with nothing arriving, not a total
    # budget — a long generation streams continuously. At 600s, retried four
    # times, a stuck request cost 40 minutes of complete silence.
    "request_timeout": 120,
    # Transient failures only (connection drops, 429, 5xx). A 4xx is a bad
    # request and retrying it just burns time and tokens.
    "max_retries": 3,

    # ── Local llama-server ──────────────────────────────────────────────────
    "llama_model_path": "",          # path to the .gguf
    "llama_server_bin": "llama-server",
    "llama_host": "127.0.0.1",
    "llama_port": 8080,
    "llama_ctx": 0,                  # 0 → computed from the fit math in models.py
    # --jinja is required for native tool_calls; -fa is required for any KV
    # quantization. This string is passed straight through to
    # --cache-type-k/--cache-type-v, so it is also the TurboQuant switch:
    #   q4_0            stock llama.cpp, ~4x, no outlier protection (default)
    #   q8_0            stock, ~2x, safest
    #   turbo3/turbo4   TurboQuant (WHT rotation + optimal scalar quant).
    #                   Needs a build that supports it — not upstream yet.
    #                   See DESIGN.md §5.2.
    # The default must stay a stock type or the harness breaks out of the box.
    "llama_kv_quant": "q4_0",
    "llama_flash_attn": True,
    "llama_extra_args": "",          # free-form pass-through (e.g. speculative flags)
    "llama_autostart": True,

    # ── Efficiency (DESIGN.md §5) ───────────────────────────────────────────
    # Prefix checkpointing: save the KV of the stable prefix once, restore on
    # every later session instead of reprefilling.
    #
    # DEFAULT OFF: unproven, not useless. llama-server's in-memory prefix cache
    # already handles repeats within a server lifetime (measured 339x on stock
    # b10242), which is the common case; this only adds value across restarts,
    # and each checkpoint costs 79-121 MB on disk. See DESIGN.md 5.3.
    "prefix_checkpoint": False,
    # One-time startup probe that the server really does reuse a repeated
    # prefix. Costs one short prefill; catches a forked build that silently
    # makes every turn 10-300x slower. See DESIGN.md 5.1.
    "check_prefix_cache": True,

    # ── Agent loop ──────────────────────────────────────────────────────────
    "max_turns": 100,                # tool-call rounds per user message
    # Rounds the provider may use to finish a reply that exceeded the cap.
    # Generous on purpose: a model that writes a lot should not be truncated
    # into failure by a limit the harness chose.
    "max_continuations": 8,
    # A second, smaller budget for the loop-level retry. Distinct from the one
    # above because the provider has already spent that; reusing the key would
    # silently double the cap. This one only helps when compaction can free
    # window before the reply resumes (DESIGN.md 5.3g).
    "max_truncation_resumes": 2,
    # Nudges after a completely empty assistant turn. Small reasoning models
    # burn the turn inside <think> and emit nothing; without this the loop
    # reads that as "done" and stops with the task half-finished.
    "max_empty_retries": 2,
    # Directories the file tools may touch beyond the working directory. The
    # jail is on Read/Write/Edit/Glob/Grep; Bash is not confined, because
    # confining a shell means parsing arbitrary commands, which does not work.
    # If that matters, run the harness in a container.
    "extra_roots": [],
    "accept_all": False,             # never prompt for permission
    "checkpoints": True,             # snapshot the tree after each change
    "recover_text_tool_calls": True,
    # Reasoning models emit a chain of thought. It is never stored in history
    # (see provider.ThinkFilter); this only controls whether it streams to the
    # screen. Hidden by default — "/think" expands the last one either way.
    "show_thinking": False,
    "transcript": True,              # plain-text log of each session, beside it

    # ── Context ─────────────────────────────────────────────────────────────
    # ── Everything below is a SHARE of the context window ───────────────────
    # llama_ctx is the only token quantity you set. Every other budget is a
    # fraction of it, so changing model or window rescales the whole harness
    # coherently. Absolute settings could silently disagree with the window —
    # a 1500-token reply cap against a 16k window produced three truncations
    # per task and looked like a harness defect until the cause was found.
    "compact_at": 0.85,              # compact once the conversation passes this
    # ...and compact down to this fraction of the budget, not to the budget.
    # Trigger and target being equal is a ratchet: every turn compacts. The gap
    # is what buys several turns between compactions, and with them the prefix
    # cache (DESIGN.md 5.1).
    "compact_to": 0.6,
    # Shares of the window, not token counts. A count that is right for a 64k
    # window is wrong for a 16k one, and every absolute here was a bug waiting
    # for someone to change llama_ctx.
    #
    # There is deliberately no `reply_share`. It used to do two jobs: reserve
    # window for the answer, and cap what the model could generate. Only the
    # first is a memory quantity, and even that was wrong — see history_budget.
    # As a generation cap it bounded nothing in aggregate (8 continuations of a
    # 32,768-token cap allow ~295k tokens in one turn, more than twice a 128k
    # window) while creating every seam the continuation machinery then had to
    # repair. Measured: at a 655-token cap the model burned all 8 rounds and
    # produced no tool call at all in 2 of 3 turns. Absent here, `max_tokens` is
    # not sent and the model generates until it is finished or the context ends.
    #
    # It survives as an *explicit* argument for mechanical calls that must be
    # bounded by definition — see summarise_span, where a note longer than the
    # span it replaces would reclaim nothing.
    "repo_map_share": 0.05,          # the per-turn focus map
    # Any single tool result. At 0.25 one result was 45% of the whole history
    # budget, so two Reads filled the conversation and no amount of hysteresis
    # could help. Truncating one result is far cheaper than compacting: the
    # model can re-read a range, whereas compaction destroys the working set
    # and the KV cache together.
    "tool_output_share": 0.08,
    "repo_map": True,

    # ── Research sub-loop ───────────────────────────────────────────────────
    "research_max_calls": 40,
}


SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty")


def _coerce(default: Any, raw: str) -> Any:
    """Coerce a CLI/REPL string to the type of its default."""
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


def load() -> dict[str, Any]:
    """Return DEFAULTS overlaid with the config file. Never raises."""
    cfg = dict(DEFAULTS)
    if tomllib is not None and CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "rb") as f:
                cfg.update(tomllib.load(f))
        except Exception:
            pass  # a corrupt config should not stop the agent from starting
    return cfg


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def save(cfg: dict[str, Any]) -> None:
    """Persist only the keys that differ from DEFAULTS."""
    HOME.mkdir(parents=True, exist_ok=True)
    lines = ["# miniharness config — only non-default values are written\n"]
    for k in sorted(cfg):
        if k.startswith("_"):
            continue  # runtime-only state
        if k in DEFAULTS and cfg[k] == DEFAULTS[k]:
            continue
        lines.append(f"{k} = {_fmt(cfg[k])}\n")
    CONFIG_PATH.write_text("".join(lines), encoding="utf-8")


def set_value(cfg: dict[str, Any], key: str, raw: str) -> Any:
    """Set one key from a string, coercing to the default's type. Returns the value."""
    if key not in DEFAULTS:
        raise KeyError(f"unknown config key: {key}")
    if key in SAMPLING_KEYS:
        # "" or "default" hands the choice back to the model card.
        raw = raw.strip()
        val = "" if raw.lower() in ("", "default", "model") else (
            int(float(raw)) if key == "top_k" else float(raw))
        cfg[key] = val
        save(cfg)
        return val
    val = _coerce(DEFAULTS[key], raw)
    cfg[key] = val
    save(cfg)
    return val
