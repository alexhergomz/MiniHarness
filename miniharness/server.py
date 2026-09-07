"""llama-server lifecycle and prefix checkpointing.

This file builds an argv and manages a subprocess. That is the whole contract
with the inference layer — MiniHarness targets **stock llama.cpp** and does not
maintain a fork. Two unmerged forks is the state this project is escaping from
(DESIGN.md §5.4); an inference trick earns its place by being a flag we pass or
~80 lines we own.

The KV configuration defaults to ``-fa -ctk q4_0 -ctv q4_0``, cutting KV
footprint ~4x on a stock build. ``llama_kv_quant`` is passed through verbatim,
so pointing ``llama_server_bin`` at a TurboQuant-capable build and setting
``llama_kv_quant=turbo3`` gets WHT-rotated KV with no change to this file.
Keeping the contract at "build an argv" is what makes that free (DESIGN.md §5.2).

Prefix checkpointing (DESIGN.md §5.3) is the local-appropriate form of a hot/cold
KV split: **hot** is the live conversation in VRAM, **cold** is the frozen prefix
serialized to disk. Restoring it makes a new session's first token near-instant
instead of reprefilling 5-15 k tokens. Everything here is best-effort — a failed
checkpoint costs a reprefill, never a crash.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time

from . import net as requests

from .config import HOME

SLOT_DIR = HOME / "slots"
LOG_PATH = HOME / "llama-server.log"


def base_url(config: dict) -> str:
    """Where the local server is. Falls back to the defaults for missing keys.

    Indexing straight into the config made this crash on any partial dict —
    every caller had to carry the full DEFAULTS or get a KeyError from a
    function that has an obvious answer for what it was asked.
    """
    from .config import DEFAULTS
    host = config.get("llama_host") or DEFAULTS["llama_host"]
    port = config.get("llama_port") or DEFAULTS["llama_port"]
    return f"http://{host}:{port}"


def live_ctx(config: dict, timeout: float = 2.0) -> int | None:
    """Context the running server is actually serving, or None.

    The config value is an intention; this is the truth. They diverge the moment
    anyone changes llama_ctx on a running server — and the harness itself
    suggests exactly that when a start fails for want of VRAM. Every budget is a
    share of the window, so a config claiming 64k against a server serving 32k
    puts every request roughly 10k tokens over the real limit.
    """
    try:
        r = requests.get(f"{base_url(config)}/slots", timeout=timeout)
        if r.status_code != 200:
            return None
        slots = r.json()
        slot = slots[0] if isinstance(slots, list) and slots else slots
        n = int(slot.get("n_ctx") or 0)
        return n or None
    except Exception:
        return None


def is_up(config: dict, timeout: float = 1.5) -> bool:
    try:
        r = requests.get(f"{base_url(config)}/health", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


# ── Argv ────────────────────────────────────────────────────────────────────
def build_args(config: dict) -> list[str]:
    """The llama-server command line. Stock flags only."""
    model_path = config.get("llama_model_path", "")
    if not model_path:
        raise ValueError("llama_model_path is not set. Run /models to pick one.")

    args = [
        config.get("llama_server_bin", "llama-server"),
        "--model", model_path,
        "--host", str(config["llama_host"]),
        "--port", str(config["llama_port"]),
        # Required for native tool_calls: without --jinja the model's chat
        # template is not applied and it cannot emit the tool_calls field.
        "--jinja",
        # Offload everything it can; llama.cpp caps this at what fits.
        "--n-gpu-layers", "999",
        # One conversation, one slot. Left on auto, llama-server picks
        # n_parallel=4 and scatters consecutive turns across slots, so the
        # prefix KV a turn just built is not the one the next turn lands on —
        # which quietly defeats the prefix caching in DESIGN.md §5.1. It also
        # makes slot 0 the only slot, which is what prefix checkpointing
        # save/restore addresses.
        "--parallel", "1",
    ]

    if config.get("llama_flash_attn", True):
        args += ["--flash-attn", "on"]
    if (kv := config.get("llama_kv_quant")):
        # KV quantization needs FlashAttention to be on.
        args += ["--cache-type-k", kv, "--cache-type-v", kv]

    ctx = int(config.get("llama_ctx") or 0)
    if ctx > 0:
        args += ["--ctx-size", str(ctx)]

    if config.get("prefix_checkpoint", True):
        SLOT_DIR.mkdir(parents=True, exist_ok=True)
        args += ["--slot-save-path", str(SLOT_DIR)]

    if (extra := config.get("llama_extra_args", "").strip()):
        args += shlex.split(extra)
    return args


def suggested_ctx(config: dict) -> int:
    """Context size the fit math allows for the configured model, or 0.

    Uses the catalog entry matching the GGUF filename. Unknown models get 0,
    meaning "let llama.cpp use the model's own default".
    """
    from .models import DEFAULT_KV_DIV, CATALOG, detect_hardware, max_context_k

    name = os.path.basename(config.get("llama_model_path", "")).lower()
    spec = next((m for m in CATALOG if m.key.split("-")[-1] in name
                 and m.family.lower().replace(".", "") in name.replace(".", "")), None)
    if spec is None:
        return 0
    budget = detect_hardware().budget_gb
    if not budget:
        return 0
    try:
        size_gb = os.path.getsize(config["llama_model_path"]) / 1e9
    except OSError:
        return 0
    kv_div = DEFAULT_KV_DIV if config.get("llama_kv_quant") else 1.0
    ctx_k = max_context_k(budget, spec, size_gb, kv_div)
    # Round down to a multiple of 1024 and cap at the model's native window.
    return int(min(ctx_k, spec.max_ctx_k) * 1024) // 1024 * 1024


# ── Lifecycle ───────────────────────────────────────────────────────────────
_PROC: subprocess.Popen | None = None


def start(config: dict, wait: float = 180.0) -> bool:
    """Start llama-server and block until /health is green. False on failure."""
    global _PROC
    if is_up(config):
        return True
    args = build_args(config)
    HOME.mkdir(parents=True, exist_ok=True)
    try:
        log = open(LOG_PATH, "ab")
        _PROC = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    except FileNotFoundError:
        raise RuntimeError(
            f"{args[0]!r} not found. Install llama.cpp, or set the path with:\n"
            f"  /config llama_server_bin=/path/to/llama-server"
        ) from None

    deadline = time.time() + wait
    while time.time() < deadline:
        if _PROC.poll() is not None:
            raise RuntimeError(_explain_exit(config))
        if is_up(config):
            return True
        time.sleep(0.5)
    return False


# Known startup failures, mapped to something the user can act on. "see the log"
# is a non-answer when the log is 900 lines of tensor allocations.
_EXIT_HINTS = [
    ("cudaMalloc failed: out of memory",
     "The GPU ran out of memory. Other processes (desktop, browser) may be "
     "holding VRAM. Try: /config llama_ctx=8192, or offload fewer layers with "
     "/config llama_extra_args='--n-gpu-layers 20'."),
    ("failed to allocate compute buffers",
     "Not enough VRAM for the compute buffers. Try a smaller batch: "
     "/config llama_extra_args='--ubatch-size 256'."),
    ("unable to load model",
     "The model file could not be loaded — check llama_model_path points at a "
     "valid .gguf."),
    ("error while handling argument",
     "llama-server rejected an argument. If you set llama_kv_quant to a "
     "TurboQuant type (turbo3/turbo4), this build does not support it."),
    ("failed to apply template",
     "The model's chat template could not be applied. It may not support tool "
     "calling; try a different model."),
]


def _explain_exit(config: dict) -> str:
    """Turn a dead llama-server into an actionable message."""
    try:
        tail = LOG_PATH.read_text(errors="replace")[-8000:]
    except OSError:
        tail = ""
    for needle, hint in _EXIT_HINTS:
        if needle in tail:
            return f"llama-server failed to start: {hint}\n(full log: {LOG_PATH})"
    last = [ln for ln in tail.splitlines() if ln.strip()][-1:] or [""]
    return f"llama-server exited during startup: {last[0][:200]}\n(full log: {LOG_PATH})"


def stop() -> None:
    global _PROC
    if _PROC and _PROC.poll() is None:
        _PROC.terminate()
        try:
            _PROC.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _PROC.kill()
    _PROC = None


def ensure(config: dict) -> bool:
    """Make sure a server is reachable, starting one if configured to."""
    if is_up(config):
        return True
    if not config.get("llama_autostart", True):
        return False
    if not config.get("llama_model_path"):
        return False
    if not config.get("llama_ctx"):
        if (ctx := suggested_ctx(config)):
            config["llama_ctx"] = ctx
    return start(config)


# ── Prefix checkpointing ────────────────────────────────────────────────────
def prefix_key(system: str, schemas: list[dict], model_path: str) -> str:
    """Identity of the cacheable prefix.

    Hashing the exact bytes is what makes invalidation automatic: change the
    system prompt, the tool set, or the model, and the key changes, so a stale
    checkpoint can never be restored onto a prefix it doesn't match.
    """
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(json.dumps(schemas, sort_keys=True).encode("utf-8"))
    h.update(os.path.basename(model_path).encode("utf-8"))
    return h.hexdigest()[:16]


def _slot_action(config: dict, action: str, filename: str, slot: int = 0) -> bool:
    try:
        r = requests.post(
            f"{base_url(config)}/slots/{slot}",
            params={"action": action},
            json={"filename": filename},
            timeout=120,
        )
        return r.status_code == 200
    except requests.RequestException:
        return False


def restore_prefix(config: dict, key: str, slot: int = 0) -> bool:
    """Load a saved prefix KV into a slot. False if there's no checkpoint."""
    if not config.get("prefix_checkpoint", True):
        return False
    if not (SLOT_DIR / f"{key}.bin").exists():
        return False
    return _slot_action(config, "restore", f"{key}.bin", slot)


def save_prefix(config: dict, key: str, slot: int = 0) -> bool:
    """Serialize a slot's KV to disk under ``key``."""
    if not config.get("prefix_checkpoint", True):
        return False
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    return _slot_action(config, "save", f"{key}.bin", slot)


def warm_prefix(config: dict, system: str, schemas: list[dict]) -> str:
    """Get the stable prefix into the server's KV cache, cheaply if possible.

    Restores an existing checkpoint when one matches; otherwise prefills the
    prefix once with a 1-token generation and saves the result for next time.
    Returns a short status string for the UI. Never raises.
    """
    if not config.get("prefix_checkpoint", True):
        return ""
    # Slot save/restore is a llama.cpp endpoint; meaningless against a cloud
    # provider or Ollama.
    from .provider import split_model
    if split_model(config.get("model", ""))[0] != "local":
        return ""
    if not is_up(config):
        return ""

    key = prefix_key(system, schemas, config.get("llama_model_path", ""))
    if restore_prefix(config, key):
        return "prefix cache restored"

    try:
        from .provider import to_openai_tools
        requests.post(
            f"{base_url(config)}/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": "ok"}],
                "tools": to_openai_tools(schemas),
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=600,
        )
    except requests.RequestException:
        return ""
    return "prefix cached" if save_prefix(config, key) else ""


def check_prefix_cache(config: dict, probe_tokens: int = 400) -> tuple[bool | None, str]:
    """Verify the server actually reuses a repeated prompt prefix.

    Sends the same short prompt twice and reads llama.cpp's own ``prompt_n``
    (tokens *evaluated*). On a working build the second request evaluates a
    handful of tokens instead of the whole prompt.

    This exists because a TurboQuant+TriAttention fork silently pinned
    ``prompt_n`` at the full prompt length on every request — no error, no
    warning, just a ~300x slowdown on every turn forever. Stock b10242 collapses
    it to 4. A harness whose headline efficiency claim is prefix stability
    (DESIGN.md 5.1) should not take that on faith.

    Returns (ok, message). ok is None when it could not be determined.
    """
    if not is_up(config):
        return None, "server not reachable"
    # Distinctive filler so we don't collide with a real conversation's prefix.
    probe = "MiniHarness prefix-cache probe.\n" + ("probe line.\n" * probe_tokens)
    body = {"model": "local",
            "messages": [{"role": "system", "content": probe},
                         {"role": "user", "content": "ok"}],
            # NOT max_tokens=1. Some builds do not commit the prompt to the slot
            # cache when only a single token is generated, so a 1-token probe
            # reports "no reuse" on a server whose cache is perfectly healthy.
            # This cost a wrong conclusion once already; 16 is enough to be sure
            # the slot is committed and still cheap.
            "max_tokens": 16, "temperature": 0}

    def once() -> int | None:
        try:
            r = requests.post(f"{base_url(config)}/v1/chat/completions",
                              json=body, timeout=600)
            r.raise_for_status()
            return (r.json().get("timings") or {}).get("prompt_n")
        except (requests.RequestException, ValueError):
            return None

    first = once()
    second = once()
    if first is None or second is None:
        return None, "server did not report timings; cannot verify prefix cache"
    if first <= 0:
        return None, "inconclusive probe"
    # A working cache re-evaluates only the last token or two.
    if second <= max(8, first * 0.1):
        return True, f"prefix cache OK ({first} -> {second} tokens re-evaluated)"
    return False, (
        f"prefix cache NOT working: a repeated prompt re-evaluated {second} of "
        f"{first} tokens. Every turn will pay full prefill (~10-300x slower). "
        f"This is usually a patched/forked llama.cpp build — try stock."
    )

