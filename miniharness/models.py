"""Plug and play: detect the hardware, pick a quant that fits, download it.

Ported from Promethean's `model_recommend.py` — the other piece of that codebase
genuinely worth keeping. The fit math is the useful part: it sizes a quant so
the model's *full context window* fits alongside the weights, accounting for the
KV cache, instead of just checking whether the weights fit and then discovering
at 40 k tokens that they don't.

Trimmed: the alternate-pick machinery that offered three variants per model.
A menu with 3 options across 8 models is a menu nobody reads. One recommendation
and one fallback per model.

Network is best-effort throughout. A failed Hugging Face fetch degrades to
"couldn't reach HF" for that repo; it never raises.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass

USER_AGENT = "miniharness"


# ── Catalog ─────────────────────────────────────────────────────────────────
@dataclass
class ModelSpec:
    key: str
    family: str
    repo: str                # Hugging Face GGUF repo
    params_b: float
    kv_gb_per_16k: float     # full-precision KV cache, GB per 16K tokens
    tier: int                # 1 = recommend first
    note: str = ""
    max_ctx_k: int = 0
    active_b: float = 0      # MoE: parameters active per token (0 = dense)
    min_gb: float = 0        # override when no small quant exists (e.g. MXFP4)
    sampling: dict | None = None   # the model card's own values; None = card is silent

    _FAMILY_MAX_CTX = {"Qwen3.5": 256, "Qwen3": 256, "Gemma 4": 128,
                       "Nemotron": 128, "GPT-OSS": 128}

    def __post_init__(self):
        if not self.max_ctx_k:
            self.max_ctx_k = self._FAMILY_MAX_CTX.get(self.family, 128)
        if self.sampling is None:
            self.sampling = FAMILY_SAMPLING.get(self.family)


# What each model's own card recommends, passed to llama-server at launch so
# they become the server's defaults for every request. The harness used to send
# temperature 0.3 on every request instead, against Qwen's documented 0.6 for
# coding — and low temperature is what drives a small model into the repetition
# loops that cost most of a day of runs (HARNESS_SPEC §2.7). The rule is not a
# better constant; it is that the harness does not overrule the model's authors.
#
# Only values a card actually documents are here. A family with no entry gets
# llama-server's own defaults rather than a guess dressed up as a recommendation.
FAMILY_SAMPLING: dict[str, dict] = {
    # Qwen3.5 / Qwen3, thinking mode, precise coding.
    "Qwen3.5": {"temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "Qwen3":   {"temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    # Reasoning on.
    "Nemotron": {"temp": 0.6, "top_p": 0.95},
    "GPT-OSS":  {"temp": 1.0, "top_p": 1.0},
}


CATALOG: list[ModelSpec] = [
    ModelSpec("qwen3.5-0.8b", "Qwen3.5", "unsloth/Qwen3.5-0.8B-GGUF", 0.8, 0.12, 2,
              "Ultra-light; phones / CPU-only."),
    ModelSpec("qwen3.5-2b", "Qwen3.5", "unsloth/Qwen3.5-2B-GGUF", 2.0, 0.28, 2,
              "Small but coherent for simple loops."),
    ModelSpec("qwen3.5-4b", "Qwen3.5", "unsloth/Qwen3.5-4B-GGUF", 4.0, 0.5, 1,
              "Good balance on <=8 GB."),
    ModelSpec("qwen3.5-9b", "Qwen3.5", "unsloth/Qwen3.5-9B-GGUF", 9.0, 1.0, 1,
              "The tuned target — best local agent on 8 GB."),
    ModelSpec("qwen3.5-27b", "Qwen3.5", "unsloth/Qwen3.5-27B-GGUF", 27.0, 2.4, 1,
              "Needs a big GPU or lots of RAM."),
    ModelSpec("gemma4-e4b", "Gemma 4", "unsloth/gemma-4-E4B-it-GGUF", 4.0, 0.45, 2,
              "Efficient 4B-class."),
    ModelSpec("gemma4-12b", "Gemma 4", "unsloth/gemma-4-12b-it-GGUF", 12.0, 1.3, 2,
              "12B; ~10 GB+ at 4-bit."),
    ModelSpec("nemotron-nano-8b", "Nemotron", "unsloth/Llama-3.1-Nemotron-Nano-8B-v1-GGUF",
              8.0, 0.95, 3, "Reliable native tool calls."),

    # Big cards. Without these the menu topped out at 27B, so a 48 GB or 80 GB
    # card was offered a model it could have run four times over. Every repo
    # here was checked to resolve on Hugging Face — a catalog entry that 404s
    # at download time fails at the worst possible moment.
    ModelSpec("qwen3-coder-30b-a3b", "Qwen3",
              "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", 30.0, 1.5, 1,
              "MoE: 30B total, 3B active — 30B quality at near-3B speed. "
              "~18 GB at 4-bit. Best pick for a 24 GB card.", active_b=3.0,
              sampling={"temp": 0.7, "top_p": 0.8, "top_k": 20,
                        "repeat_penalty": 1.05}),
    ModelSpec("qwen3-32b", "Qwen3", "unsloth/Qwen3-32B-GGUF", 32.0, 4.0, 2,
              "Dense 32B, ~20 GB at 4-bit. Stronger than the MoE per token, "
              "but far slower and its KV cache is heavy."),
    ModelSpec("gpt-oss-120b", "GPT-OSS", "unsloth/gpt-oss-120b-GGUF", 120.0, 1.1, 1,
              "MoE, ships native MXFP4 at ~63 GB — the 80 GB-card option. "
              "There is no smaller quant, so it either fits or it does not.",
              active_b=5.1, min_gb=65.0),
    ModelSpec("qwen3-235b-a22b", "Qwen3", "unsloth/Qwen3-235B-A22B-GGUF",
              235.0, 3.0, 3,
              "MoE: 235B total, 22B active. Needs ~120 GB at 4-bit — multi-GPU, "
              "or one big card with most experts on CPU.", active_b=22.0),
]
CATALOG_BY_KEY = {m.key: m for m in CATALOG}


def spec_for_path(path: str) -> ModelSpec | None:
    """The catalog entry a GGUF filename belongs to, or None.

    Longest key match wins, so `qwen3-coder-30b-a3b` is not mistaken for
    `qwen3-32b`, and `qwen3.5-4b` is not mistaken for `gemma4-e4b`.
    """
    name = os.path.basename(path).lower().replace("_", "-")
    best = None
    for m in CATALOG:
        fam = re.sub(r"[ .-]", "", m.family.lower())
        size = m.key.split("-")[-1]
        if fam not in re.sub(r"[ .-]", "", name):
            continue
        if m.family == "Qwen3" and "qwen3.5" in name:
            continue                      # "qwen3" is a prefix of "qwen3.5"
        if ("coder" in m.key) != ("coder" in name):
            continue
        if re.search(rf"(?<![0-9a-z]){re.escape(size)}(?![0-9])", name):
            if best is None or len(m.key) > len(best.key):
                best = m
    return best


# ── Quant parsing / scoring ─────────────────────────────────────────────────
_QUANT_RE = re.compile(r"(UD-)?((?:IQ|Q)\d+(?:_[A-Z0-9]+)*|BF16|F16|F32)", re.IGNORECASE)
_EXCLUDE = ("bf16", "f16", "f32")  # full precision: not agent quants


def parse_quant_label(filename: str) -> str | None:
    m = _QUANT_RE.search(filename)
    if not m:
        return None
    return f"{m.group(1) or ''}{m.group(2)}".upper()


def quant_quality(label: str) -> float:
    """Heuristic fidelity score for ranking quants (higher = better)."""
    lab = label.upper()
    digits = re.search(r"(\d+)", lab)
    score = (int(digits.group(1)) if digits else 4) * 10.0
    if "_K_M" in lab or "_K_XL" in lab:
        score += 3
    elif "_K_S" in lab:
        score += 1
    elif "_K" in lab:
        score += 2
    if lab.startswith("IQ"):
        score -= 1      # i-quants slightly below same-bit K-quants for agents
    if lab.startswith("UD-"):
        score += 1.5    # Unsloth dynamic: better quality per byte
    return score


@dataclass
class Quant:
    label: str
    size_gb: float
    filename: str = ""
    shards: int = 1


def parse_tree(tree: list) -> list[Quant]:
    """HF /tree/main listing -> deduped quants, sharded files summed."""
    agg: dict[str, dict] = {}
    for entry in tree:
        path = entry.get("path", "")
        if not path.endswith(".gguf"):
            continue
        base = path.rsplit("/", 1)[-1]
        if base.lower().startswith("mmproj"):
            continue
        label = parse_quant_label(base)
        if not label or label.lower() in _EXCLUDE:
            continue
        size = entry.get("size") or (entry.get("lfs") or {}).get("size") or 0
        a = agg.setdefault(label, {"gb": 0.0, "file": base, "shards": 0})
        a["gb"] += float(size) / 1e9
        a["shards"] += 1
        if base < a["file"]:      # shard 00001-of-... sorts first
            a["file"] = base
    out = [Quant(k, round(v["gb"], 2), v["file"], v["shards"])
           for k, v in agg.items() if v["gb"] > 0]
    out.sort(key=lambda q: q.size_gb)
    return out


def fetch_quants(repo: str, timeout: float = 8.0) -> list[Quant] | None:
    """Available GGUF quants for a repo. None if HF is unreachable."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tree = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return parse_tree(tree) if isinstance(tree, list) else None


def download_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


# ── Hardware ────────────────────────────────────────────────────────────────
@dataclass
class Hardware:
    ram_gb: float | None = None
    vram_gb: float | None = None
    gpu_name: str = ""

    @property
    def budget_gb(self) -> float | None:
        """VRAM if there's a discrete GPU, else system RAM (CPU / unified)."""
        return self.vram_gb or self.ram_gb


def _detect_ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip()) / 1e9
    except Exception:
        pass
    return None


def _vram_sysfs() -> tuple[float, str]:
    """amdgpu/i915 report VRAM in sysfs. Returns the largest card found."""
    import glob
    best, best_name = 0.0, ""
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
        try:
            with open(p) as f:
                gb = int(f.read().strip()) / 1e9
        except (OSError, ValueError):
            continue
        if gb > best:
            best = gb
            try:
                with open(os.path.join(os.path.dirname(p), "uevent")) as f:
                    for line in f:
                        if line.startswith("DRIVER="):
                            best_name = line.split("=", 1)[1].strip()
            except OSError:
                pass
    return best, best_name


def _vram_nvidia_smi() -> tuple[float, str]:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        if out.returncode == 0 and out.stdout.strip():
            mb, _, name = out.stdout.strip().splitlines()[0].partition(",")
            return int(mb.strip()) / 1024, name.strip()
    except Exception:
        pass
    return 0.0, ""


def _vram_libcuda() -> tuple[float, str]:
    """Ask the CUDA driver directly, via ctypes.

    Worth the 30 lines: `nvidia-smi` breaks on an NVML driver/library version
    mismatch, which happens routinely after a driver update, while libcuda keeps
    working. Without this the fallback chain silently reports an integrated GPU
    on any hybrid laptop whose nvidia-smi is broken.
    """
    import ctypes
    for soname in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(soname)
        except OSError:
            continue
        try:
            if lib.cuInit(0) != 0:
                return 0.0, ""
            count = ctypes.c_int()
            if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
                return 0.0, ""
            best, best_name = 0.0, ""
            for i in range(count.value):
                dev = ctypes.c_int()
                if lib.cuDeviceGet(ctypes.byref(dev), i) != 0:
                    continue
                nbytes = ctypes.c_size_t()
                if lib.cuDeviceTotalMem_v2(ctypes.byref(nbytes), dev) != 0:
                    continue
                gb = nbytes.value / 1e9
                if gb > best:
                    buf = ctypes.create_string_buffer(256)
                    name = ""
                    if lib.cuDeviceGetName(buf, 256, dev) == 0:
                        name = buf.value.decode("utf-8", "replace")
                    best, best_name = gb, name
            return best, best_name
        except (AttributeError, OSError):
            return 0.0, ""
    return 0.0, ""


def _detect_vram_gb() -> tuple[float | None, str]:
    """Largest GPU across every probe we have.

    Deliberately probes *all* sources and takes the max rather than returning
    the first hit. A hybrid laptop exposes its integrated GPU through sysfs and
    its discrete GPU only through the vendor driver, so first-hit-wins picks the
    500 MB iGPU over the 6 GB dGPU and every downstream recommendation is wrong.
    """
    best, name = 0.0, ""
    for probe in (_vram_sysfs, _vram_nvidia_smi, _vram_libcuda):
        try:
            gb, n = probe()
        except Exception:
            continue
        if gb > best:
            best, name = gb, n
    return (best, name) if best > 0 else (None, "")


def detect_hardware() -> Hardware:
    vram, name = _detect_vram_gb()
    return Hardware(ram_gb=_detect_ram_gb(), vram_gb=vram, gpu_name=name)


# ── Fit math ────────────────────────────────────────────────────────────────
RUNTIME_OVERHEAD_GB = 0.8   # compute / activation buffers
MIN_CTX_K = 8               # a quant that can't hold 8K context isn't viable
DEFAULT_KV_DIV = 4.0        # q4_0 KV cache (DESIGN.md §5.2)

# KV cache types, most faithful first, with how much smaller each is than f16.
# The recommendation picks the *least* aggressive one that still reaches the
# model's native context: spare memory is better spent on KV fidelity than on
# context the model cannot use.
KV_OPTIONS: list[tuple[str, float]] = [
    ("f16",  1.0),
    ("q8_0", 2.0),
    ("q5_1", 2.9),
    ("q4_0", 4.0),
]


def kv_gb(model: ModelSpec, ctx_k: float, kv_div: float = DEFAULT_KV_DIV) -> float:
    return model.kv_gb_per_16k * (ctx_k / 16.0) / max(kv_div, 0.01)


def max_context_k(budget_gb: float, model: ModelSpec, quant_gb: float,
                  kv_div: float = DEFAULT_KV_DIV) -> float:
    """Largest context (K tokens) that fits alongside weights + overhead.

    Capped at what the model actually supports. The arithmetic answers "what
    fits in VRAM" and nothing more, so on a well-provisioned machine it happily
    returned 9,600k for a model whose limit is 256k — and ``__main__`` sets
    ``llama_ctx = ctx_k * 1024`` from this, so the server would be started with
    a context the weights cannot serve. Free memory is a ceiling, not a
    permission.
    """
    free = budget_gb - quant_gb - RUNTIME_OVERHEAD_GB
    if free <= 0:
        return 0.0
    per_16k = model.kv_gb_per_16k / max(kv_div, 0.01)
    room = 999.0 if per_16k <= 0 else (free / per_16k) * 16.0
    return min(room, float(model.max_ctx_k))


@dataclass
class Pick:
    model: ModelSpec
    quant: Quant
    ctx_k: int
    fits_target: bool
    recommended: bool = False
    kv_quant: str = "q4_0"


def recommend_for_model(budget_gb: float, model: ModelSpec, quants: list[Quant],
                        kv_div: float = DEFAULT_KV_DIV) -> list[Pick]:
    """Best pick for this model, plus at most one fallback.

    The target is the model's native context — not less, and not more. Less
    means shipping a model that cannot do what it was trained to do; more is
    unreachable anyway, since context is capped at what the weights support.

    So KV quantization is *chosen*, not assumed. For each weight quant, take the
    least aggressive KV type that still reaches native context: on a large card
    that means f16 KV rather than q4_0 and the same context, which is free
    quality. Only when no KV type gets there does the pick fall back to the
    largest context available, and that is reported as reduced.
    """
    target = float(model.max_ctx_k)

    def best_kv(quant_gb: float) -> tuple[str, float]:
        """(kv label, context) — the most faithful KV that reaches native."""
        best_label, best_ctx = KV_OPTIONS[-1][0], 0.0
        for label, div in KV_OPTIONS:
            c = max_context_k(budget_gb, model, quant_gb, div)
            if c >= target:
                return label, c          # first (most faithful) that suffices
            if c > best_ctx:
                best_label, best_ctx = label, c
        return best_label, best_ctx

    scored = []
    for q in quants:
        kv, c = best_kv(q.size_gb)
        if c >= MIN_CTX_K:
            scored.append((q, c, kv))
    if not scored:
        return []

    fits = [t for t in scored if t[1] >= target]
    if fits:
        q, c, kv = max(fits, key=lambda t: quant_quality(t[0].label))
    else:
        q, c, kv = max(scored, key=lambda t: (t[1], quant_quality(t[0].label)))
    best = Pick(model, q, int(min(c, 9999)), c >= target, recommended=True, kv_quant=kv)

    # One fallback, framed as the fidelity <-> context trade-off: if the pick
    # already covers the full window, offer more fidelity; otherwise more context.
    rq = quant_quality(q.label)
    if best.fits_target:
        alts = [t for t in scored if quant_quality(t[0].label) > rq]
        alt = max(alts, key=lambda t: quant_quality(t[0].label)) if alts else None
    else:
        alts = [t for t in scored if quant_quality(t[0].label) < rq]
        alt = max(alts, key=lambda t: t[1]) if alts else None

    picks = [best]
    if alt:
        picks.append(Pick(model, alt[0], int(min(alt[1], 9999)),
                          alt[1] >= target, kv_quant=alt[2]))
    return picks


def _min_fit_gb(model: ModelSpec) -> float:
    """Coarse smallest-quant footprint (~3-bit), for pre-filtering.

    ``min_gb`` overrides it for models with no small quant: gpt-oss ships
    native MXFP4 and nothing below it, so the params-based guess would offer it
    on a card ~12 GB short of actually holding it.

    Footprint follows *total* parameters even for MoE — all experts are
    resident. Only speed follows ``active_b``, which is why a 30B/3B MoE is the
    better buy at the same footprint as a dense 32B.
    """
    if model.min_gb:
        return model.min_gb
    return model.params_b * 0.42 + RUNTIME_OVERHEAD_GB


def candidates(budget_gb: float) -> list[ModelSpec]:
    out = [m for m in CATALOG if _min_fit_gb(m) <= budget_gb * 1.05]
    out.sort(key=lambda m: (m.tier, -m.params_b))
    return out


def build_menu(budget_gb: float, max_models: int = 4,
               kv_div: float = DEFAULT_KV_DIV) -> list[Pick]:
    """The first-run menu: fetch quants concurrently and narrow each model."""
    import concurrent.futures as cf

    cands = candidates(budget_gb)[:max_models]
    if not cands:
        return []
    results: dict[str, list[Quant] | None] = {}
    with cf.ThreadPoolExecutor(max_workers=min(len(cands), 8)) as ex:
        futs = {ex.submit(fetch_quants, m.repo): m for m in cands}
        for fut in cf.as_completed(futs):
            m = futs[fut]
            try:
                results[m.key] = fut.result()
            except Exception:
                results[m.key] = None

    menu: list[Pick] = []
    for m in cands:
        quants = results.get(m.key)
        if quants:
            menu.extend(recommend_for_model(budget_gb, m, quants, kv_div))

    # A model that reaches its native context comes first, however small it is.
    # Ordering by size alone made the default a bigger model running at a
    # fraction of the window it was trained for — on a 6 GB card, a 9B at 151k
    # of 256k ahead of a 4B at its full 256k. Capability the weights cannot
    # exercise is not capability.
    menu.sort(key=lambda p: (not p.fits_target, p.model.tier, -p.model.params_b))
    return menu


# ── Download ────────────────────────────────────────────────────────────────
def download(repo: str, filename: str, dest_dir, progress=None) -> str:
    """Stream a GGUF to ``dest_dir``. Returns the local path.

    Resumes a partial download with a Range request; verifies the final size
    against Content-Length so a truncated file is never silently accepted.
    """
    from . import net as requests
    from pathlib import Path

    dest_dir = Path(dest_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / filename
    part = out.with_suffix(out.suffix + ".part")

    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"

    with requests.get(download_url(repo, filename), headers=headers,
                      stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have and r.status_code == 206 else "wb"
        if mode == "wb":
            have = 0
        done = have
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)

    if total and part.stat().st_size != total:
        raise RuntimeError(
            f"download incomplete: got {part.stat().st_size} of {total} bytes")
    part.rename(out)
    return str(out)
