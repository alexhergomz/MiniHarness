"""One wire format, done well: OpenAI-compatible streaming chat completions.

Everything the harness talks to speaks this — llama.cpp's ``llama-server``,
Ollama's ``/v1``, LM Studio, OpenAI, DeepSeek, Moonshot, DashScope, OpenRouter,
and Anthropic via its OpenAI-compat endpoint. Supporting exactly one wire format
is the QOI call: the alternative is 1,200 lines of per-vendor special cases.

Two things here are load-bearing for local models and must not be "simplified"
away:

1. **Truncation detection.** ``finish_reason == "length"`` is surfaced so the
   loop can apply the alternation fix (see ``loop.py``).
2. **Text tool-call recovery.** Small models write tool calls into the message
   body. Without ``toolcalls.recover_tool_calls`` those turns silently no-op.
"""

from __future__ import annotations

import json
import threading as _threading
import time as _time
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Iterable

from . import net as requests

from . import config as _cfg
from . import toolcalls

# ── Provider table ──────────────────────────────────────────────────────────
# name -> (base_url, api-key env var). A provider is a URL and a key; anything
# needing more than that does not belong here.
PROVIDERS: dict[str, tuple[str, str]] = {
    "local":      ("http://127.0.0.1:8080/v1", ""),
    "ollama":     ("http://127.0.0.1:11434/v1", ""),
    "lmstudio":   ("http://127.0.0.1:1234/v1", ""),
    "openai":     ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic":  ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "deepseek":   ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "moonshot":   ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "dashscope":  ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "custom":     ("", ""),
}

_PREFIX_HINTS = (
    ("gpt-", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
    ("claude", "anthropic"),
    ("deepseek", "deepseek"),
    ("moonshot", "moonshot"), ("kimi", "moonshot"),
    ("qwen-", "dashscope"),
)


def split_model(model: str) -> tuple[str, str]:
    """``"ollama/qwen3-coder"`` -> ``("ollama", "qwen3-coder")``.

    An explicit ``provider/name`` or ``provider:name`` prefix wins. Otherwise
    the provider is guessed from the model name, defaulting to ``local`` so a
    bare name means "whatever llama-server is running".
    """
    for sep in ("/", ":"):
        if sep in model:
            head, tail = model.split(sep, 1)
            if head in PROVIDERS:
                return head, tail
    low = model.lower()
    for prefix, provider in _PREFIX_HINTS:
        if low.startswith(prefix):
            return provider, model
    return "local", model


def resolve_endpoint(model: str, config: dict) -> tuple[str, str, str]:
    """Return (base_url, api_key, bare_model_name)."""
    provider, bare = split_model(model)
    base_url = config.get("base_url") or PROVIDERS.get(provider, ("", ""))[0]
    if provider == "local" and not config.get("base_url"):
        base_url = f"http://{config['llama_host']}:{config['llama_port']}/v1"
    if not base_url:
        raise ValueError(
            f"No base_url for provider '{provider}'. Set one with: /config base_url=http://..."
        )
    key_env = config.get("api_key_env") or PROVIDERS.get(provider, ("", ""))[1]
    api_key = os.environ.get(key_env, "") if key_env else ""
    if key_env and not api_key:
        raise ValueError(f"Provider '{provider}' needs {key_env} in the environment.")
    return base_url.rstrip("/"), api_key, bare


# ── Streamed events ─────────────────────────────────────────────────────────
@dataclass
class Continuing:
    """Emitted at the start of each continuation round, while it is happening.

    The existing "continued Nx" notice is reported *after* the turn ends, which
    is no use for the case that hurts: a turn that has not ended. Measured on
    this hardware, one round generates 8,192 tokens at ~47 tok/s = 176 s, and a
    turn may take nine of them — up to 26 minutes, longer at large context, with
    nothing emitted the entire time. That is indistinguishable from a hang, and
    it was diagnosed as one twice.
    """
    round: int
    rounds: int
    elapsed: float


@dataclass
class StoppedCircling:
    """Deliberation was ended by the harness, with the reasoning preserved.

    A distinct event rather than a Notice: everything `stream_complete` yields
    that is not a chunk is read as the finished turn by the loop, so a bare
    Notice here would be mistaken for one.
    """
    reason: str


@dataclass
class TextChunk:
    text: str


@dataclass
class ThinkChunk:
    """Reasoning output. Shown while streaming, never stored in history."""
    text: str


@dataclass
class ToolDraft:
    """A tool call is being written: its name, and how much of it so far.

    Writing a file means streaming its whole content as the call's arguments,
    which produces no text and no reasoning — so the screen used to freeze on
    the last "thinking…" for as long as that took. Watched: "thought for 55s ·
    ~177 tokens", where the thinking was two seconds and the rest was a file
    being written. Progress only; the finished call still arrives in the turn.
    """
    name: str
    chars: int


class ThinkFilter:
    """Split a token stream into visible text and <think> reasoning.

    Local reasoning models (Qwen3, R1 distills, and friends) emit their chain of
    thought inline in ``content`` rather than in a separate field. Left alone it
    lands in the conversation history, where it costs context on every
    subsequent turn and gives the model its own half-formed reasoning to re-read.

    Tags can straddle chunk boundaries, so a partial tag is held back rather
    than emitted and corrected later.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.in_think = False
        self._buf = ""

    def feed(self, text: str) -> tuple[str, str]:
        """Return (visible, thinking) for this chunk."""
        self._buf += text
        visible, thinking = [], []
        while self._buf:
            tag = self.CLOSE if self.in_think else self.OPEN
            idx = self._buf.find(tag)
            if idx >= 0:
                head, self._buf = self._buf[:idx], self._buf[idx + len(tag):]
                (thinking if self.in_think else visible).append(head)
                self.in_think = not self.in_think
                continue
            # No complete tag. Hold back anything that could still become one.
            keep = _partial_tail(self._buf, tag)
            emit, self._buf = self._buf[:len(self._buf) - keep], self._buf[len(self._buf) - keep:]
            if emit:
                (thinking if self.in_think else visible).append(emit)
            break
        return "".join(visible), "".join(thinking)

    def flush(self) -> tuple[str, str]:
        """Emit whatever is still buffered at end of stream."""
        rest, self._buf = self._buf, ""
        return ("", rest) if self.in_think else (rest, "")


def _partial_tail(buf: str, tag: str) -> int:
    """Length of the suffix of ``buf`` that is a prefix of ``tag``."""
    for n in range(min(len(tag) - 1, len(buf)), 0, -1):
        if buf.endswith(tag[:n]):
            return n
    return 0


@dataclass
class AssistantTurn:
    """The final assembled assistant message."""
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # {"id","name","input"}
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    thinking: str = ""   # kept for display/debug; deliberately not in to_message()
    # How many extra rounds were needed to finish this reply past the cap.
    # Reported so continuation cannot quietly multiply wall time unnoticed.
    continuations: int = 0

    @property
    def truncated(self) -> bool:
        # "network" is treated like "length": the response stopped early through
        # no fault of the model, and the right recovery is the same — keep the
        # partial turn, keep alternation, ask it to continue.
        return self.finish_reason in ("length", "network")

    def to_message(self) -> dict:
        """Serialize to an OpenAI-format assistant message for the history.

        Reasoning rides along *only* when the message carries tool calls. An
        assistant turn is not finished until its tool calls have been answered,
        and the model needs what it worked out in order to read the results —
        without it, verified, the request for the next step contained nothing it
        had concluded, so it re-derived its plan every turn.

        Measured against this server: reasoning_content on a message with tool
        calls is rendered and used (the model recalled a value stated only
        there); on a completed message it is ignored, and an inline <think> tag
        in content is ignored too. So it is attached exactly where the template
        will honour it and nowhere else.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls and (self.thinking or "").strip():
            msg["reasoning_content"] = self.thinking.strip()
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["input"], sort_keys=True),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


def to_openai_tools(schemas: Iterable[dict]) -> list[dict]:
    """Convert internal tool schemas to OpenAI function-tool format.

    Order is preserved exactly as given. This matters: reordering tools changes
    the prompt prefix and invalidates the KV cache (DESIGN.md §5.1).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


# ── Connect, with bounded retries ───────────────────────────────────────────
# Retry only these: transient by definition. A 400/401/404 is a bug in the
# request and retrying it just wastes the user's time and money.
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _connect(url: str, headers: dict, payload: dict, config: dict):
    """POST and return a streaming response, retrying transient failures.

    Retries happen here, *before* any token has been handed to the caller.
    Once the stream has yielded output there is no safe retry — replaying would
    duplicate text the user already saw — so mid-stream failures are handled by
    the caller as a truncated turn instead.
    """
    attempts = max(0, int(config.get("max_retries", 3)))

    # This is an *idle* bound, not a total one: it is the socket timeout, so it
    # limits how long we wait with nothing arriving. A healthy stream sends a
    # token every few milliseconds, so a tight value costs a long generation
    # nothing while a stuck one is caught quickly.
    #
    # It was 600 s, retried four times: 40 minutes of complete silence before
    # anything was reported. Three long-horizon runs were investigated as hangs
    # on the strength of that silence, and the harness said nothing throughout —
    # the same failure as an unannounced continuation, in a different layer.
    timeout = config.get("request_timeout", 120)
    last = ""
    for attempt in range(attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 stream=True, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            last = f"{type(e).__name__}: {e}"
            if attempt >= attempts:
                raise RuntimeError(f"could not reach {url}: {last}") from e
            _note(config, f"no response in {timeout}s — retry "
                          f"{attempt + 1}/{attempts}")
        else:
            if resp.status_code < 400:
                return resp
            body = resp.text[:400]
            if resp.status_code not in RETRY_STATUS or attempt >= attempts:
                raise RuntimeError(f"{resp.status_code} from {url}: {body}")
            last = f"HTTP {resp.status_code}: {body}"
            # Honour Retry-After when the server sends one (rate limits do).
            try:
                hinted = float(resp.headers.get("Retry-After", ""))
            except ValueError:
                hinted = 0.0
            time.sleep(max(hinted, _backoff(attempt)))
            continue
        time.sleep(_backoff(attempt))
    raise RuntimeError(f"could not reach {url}: {last}")  # pragma: no cover


class _Heartbeat:
    """Report that a request is still outstanding, while it is outstanding.

    Every silent-stall investigation in this project failed for the same
    reason: the blocking call says nothing until it returns, so a request that
    never returns leaves no trace at all. Four separate causes were proposed
    and disproved from timing alone. A thread that ticks while the call is in
    flight costs nothing and turns "it hung" into "it hung waiting for X".
    """

    def __init__(self, config: dict, what: str, every: float = 30.0):
        self._config, self._what, self._every = config, what, every
        self._done = _threading.Event()
        self._last = _time.monotonic()
        self._thread = _threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._done.wait(self._every):
            idle = _time.monotonic() - self._last
            if idle >= self._every:
                _note(self._config,
                      f"still waiting on {self._what} — {idle:.0f}s with nothing "
                      f"arriving")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()

    def saw_data(self):
        """Reset the clock: a stream that is delivering is not stalled."""
        self._last = _time.monotonic()


def _note(config: dict, text: str) -> None:
    """Report something the user would otherwise experience as a hang.

    A callback rather than an event, because `_connect` runs before the
    generator has yielded anything and has no channel of its own.
    """
    sink = config.get("_notify")
    if callable(sink):
        try:
            sink(text)
        except Exception:
            pass


def _backoff(attempt: int) -> float:
    """0.5s, 1s, 2s, ... capped. No jitter: single-user, no thundering herd."""
    return min(0.5 * (2 ** attempt), 8.0)


# ── Streaming ───────────────────────────────────────────────────────────────
CONTINUE = ("Continue your previous message from exactly where it stopped. "
            "Do not repeat anything you already wrote, and do not summarise.")

CONCLUDE = (
    "You have reached the limit of the space available for reasoning. Stop "
    "deliberating and act now: make your next tool call, or give your answer "
    "using what you have already worked out above."
)

CONTINUE_THINKING = (
    "That is your reasoning so far. It was cut off by a length limit, not "
    "finished. Carry on from exactly where it stopped — do not start over and "
    "do not repeat it — then give your answer or make your tool call."
)


def _room_to_continue(messages: list[dict], system: str, written: str,
                      tool_schemas: list[dict], config: dict) -> bool:
    """Is there window left for another continuation round?

    Each round resends everything written so far, and the loop compacts *before*
    this call rather than during it. Without a bound, a reply long enough to
    matter would grow the request past the window and 400 mid-turn — converting
    "the answer was long" into "the turn failed", which is the exact penalty
    continuation exists to remove.

    Uses the same budget as compaction, which already reserves room for the
    schemas, the focus map and the next reply.
    """
    from . import config as _cfg
    from .context import estimate_tokens

    if not config.get("llama_ctx"):
        return True                    # window unknown: only the round cap applies
    used = estimate_tokens(messages, system) + len(written) // 4
    return used <= _cfg.history_budget(config, tool_schemas)


def _conclude(model, system, messages, reasoned, tool_schemas, config):
    """One final round: reasoning kept, deliberation ended.

    Reached only when the window ran out with everything spent on thinking and
    nothing on an answer. Without it the turn ends empty and the loop retries
    from scratch, paying the same cost to reach the same place.

    The reasoning must be trimmed to fit. This is called *because* the window
    bound tripped, so by construction history + reasoning is already over
    budget: sending it whole would 400 — the landing round would become the
    very failure it exists to avoid. The tail is kept, because the end of a
    chain of thought is where its conclusions are; the model is told the front
    was cut so it does not treat the remainder as the whole argument.
    """
    from . import config as _cfg
    from .context import estimate_tokens

    room = _cfg.history_budget(config, tool_schemas) - estimate_tokens(messages, system)
    if config.get("llama_ctx") and len(reasoned) // 4 > room:
        keep = max(0, room * 4)
        reasoned = ("[earlier reasoning trimmed to fit the context window]\n\n"
                    + reasoned[-keep:]) if keep else ""
    request = messages + [
        {"role": "assistant", "content": reasoned},
        {"role": "user", "content": CONCLUDE},
    ] if reasoned.strip() else messages
    # The landing round is watched too.
    #
    # It was not, and that was the one place it mattered most: this is reached
    # *because* the model was already circling, so an unbounded call here is
    # the worst case, not an edge case. Measured — a landing round ran to
    # 31,191 tokens over 31 minutes while the round loop's own bound of 13,107
    # sat one function away, unused.
    #
    # There is no second landing round to escalate to, so on a trigger the
    # partial turn is returned and marked truncated: the loop's empty-turn and
    # continuation handling take it from there, with the reasoning intact.
    watch = _Deliberation(config)
    out = None
    text_seen: list[str] = []
    think_seen: list[str] = []
    # And it is asked to act with thinking switched off. It used to leave
    # thinking on, and the model answered "stop deliberating" by opening a new
    # <think> block and deliberating again — until this round's own guard
    # tripped, the loop continued the partial turn, and the cycle repeated.
    # Watched live: past the limit and still thinking, several rounds on. The
    # reasoning so far is in the request, so nothing is lost by this; what is
    # removed is only the option to keep putting the decision off.
    landing = dict(config, disable_thinking=True)
    try:
        for event in stream(model, system, request, tool_schemas, landing):
            if isinstance(event, ToolDraft):
                yield event
                continue
            if isinstance(event, (TextChunk, ThinkChunk)):
                yield event
                if isinstance(event, ThinkChunk):
                    think_seen.append(event.text)
                    if watch.feed(event.text):
                        break
                else:
                    text_seen.append(event.text)
            else:
                out = event
    except RuntimeError:
        return None
    if out is None and (text_seen or think_seen):
        out = AssistantTurn(text="".join(text_seen), finish_reason="length",
                            thinking="".join(think_seen))
    return out


class _Deliberation:
    """Watches reasoning as it streams and says when it has stopped going anywhere.

    Qwen documents this failure in its own model card: the model has no strong
    stop-thinking signal, so it "rephrases the same logic and revisits the same
    conclusions" until something cuts it off. Measured here, one turn ran to
    45,101 tokens; another circled for 13,000 while writing "Actually, I think
    I've been going in circles."

    Two triggers, because one is not enough:

    * **Repetition.** The honest signal, and the one that fires early. Verbatim
      repeats are only half of it — with the DRY sampler on, the model stopped
      repeating and started paraphrasing, and the distinct-line ratio still sat
      at 0.41 against 0.72+ for a healthy turn. So the ratio is what is
      measured, not exact matches.
    * **Sheer length.** A backstop for a loop novel enough to keep its ratio up.
      Deliberately generous: healthy turns here run a few hundred to ~2,000
      tokens of reasoning, so the bound sits an order of magnitude above normal
      work and only catches something that has plainly stopped converging.

    Neither truncates the model. Both hand over to the landing round, which
    keeps every token of reasoning and asks for a decision (§2.3, §3.4).
    """

    MIN_SAMPLE = 12000       # characters before the ratio means anything: one window
    CHECK_EVERY = 2000
    MIN_DISTINCT = 0.40      # measured: 0.019 looping, 0.41 paraphrasing, 0.72+ healthy
    WINDOW = 12000           # characters — the last ~3,000 tokens
    MAX_COMPRESSION = 0.08   # measured: healthy 0.24-0.43, loops 0.003-0.022

    # Measured with the model's own tokenizer on 268,089 characters of its
    # reasoning: 70,611 tokens. Python source runs 4.06.
    CHARS_PER_TOKEN = 3.8

    def __init__(self, config: dict):
        # The backstop is the model's own number, not a share of the window.
        # Qwen3.5's card: "We recommend using an output length of 32,768 tokens
        # for most queries" (81,920 for competition-grade maths and code). It
        # was 8,192 and then 16,384 — sized from the harness's side, and short
        # of what the model's authors say it needs. It can be this generous
        # because it is no longer the guard that does the work: circling is
        # caught by the repetition check over recent reasoning, which fired at
        # 4,000-7,000 tokens on every circle of a live build. This only stops
        # reasoning that is new line after line and still never lands.
        #
        # Reasoning is dropped by the template at the next user message, but
        # not before: within a turn every step's reasoning stays in context, so
        # a long think costs room for the rest of that turn. Compaction pays
        # for that; the window's own bound still ends a round that would not
        # fit. Neither is a reason to cut the model short of its card.
        limit = int(config.get("think_limit", 32768))
        self.cap_chars = int(limit * self.CHARS_PER_TOKEN)
        self.enabled = bool(config.get("stop_circling", True))
        self._buf: list[str] = []
        self._len = 0
        self._checked = 0

    def feed(self, text: str) -> str | None:
        if not self.enabled or not text:
            return None
        self._buf.append(text)
        self._len += len(text)
        if self._len >= self.cap_chars:
            return (f"reasoning reached ~{int(self._len / self.CHARS_PER_TOKEN):,} "
                    f"tokens without reaching a conclusion")
        if self._len - self._checked < self.CHECK_EVERY or self._len < self.MIN_SAMPLE:
            return None
        self._checked = self._len
        # Measured over the *recent* reasoning, not all of it. The ratio used
        # to be taken over everything since the first line, so a healthy start
        # kept it up long after the model had begun going round: a saved
        # 67,000-token turn read 0.42-0.48 cumulatively — just over the line —
        # while its most recent 3,000 tokens were 0.26 distinct by line and
        # 0.20 by phrase. It had decided the same fix 38 times without making
        # it. Over a window, that trips at ~7,000 tokens instead of never.
        recent = "".join(self._buf)[-self.WINDOW:]
        # How well the recent reasoning compresses, independent of lines and
        # words. Watched on a live build: the model quoted a traceback's caret
        # line and wrote ~100,000 "^" on a single line for eighteen minutes.
        # That is one line (below the line count below) and one "word" (no
        # 8-word phrases at all), so both ratios were blind to the simplest
        # loop there is. zlib is not: varied text shrinks to 0.24-0.43 of its
        # size (measured over this repository's code and prose, and the
        # model's healthy reasoning), a circling runaway to 0.022, the caret
        # line to 0.003. 0.08 leaves a wide margin either side.
        import zlib
        raw = recent.encode("utf-8", "replace")
        packed = len(zlib.compress(raw, 6)) / max(1, len(raw))
        if packed < self.MAX_COMPRESSION:
            return (f"reasoning stopped progressing — the last "
                    f"{int(len(recent) / self.CHARS_PER_TOKEN):,} tokens compress "
                    f"{1 / max(packed, 1e-6):.0f}× (varied text: about 3×)")
        lines = [l.strip() for l in recent.splitlines() if l.strip()]
        if len(lines) < 20:
            return None
        line_ratio = len(set(lines)) / len(lines)
        # Both measures, so quoting one block of code twice — repeated lines,
        # but surrounded by new prose — is not mistaken for circling.
        words = recent.split()
        shingles = [" ".join(words[i:i + 8]) for i in range(len(words) - 8)]
        phrase_ratio = len(set(shingles)) / len(shingles) if shingles else 1.0
        if line_ratio < self.MIN_DISTINCT and phrase_ratio < self.MIN_DISTINCT:
            return (f"reasoning stopped progressing — over the last "
                    f"{int(len(recent) / self.CHARS_PER_TOKEN):,} tokens only {line_ratio:.0%} of lines "
                    f"and {phrase_ratio:.0%} of phrases were new")
        return None


def _resume_request(messages: list[dict], written: str, reasoned: str) -> list[dict]:
    """The request for a continuation round: resume, never restart.

    Three cases, in priority order.

    Visible text exists — feed it back as the assistant's own words and ask it
    to carry on. Previously the remainder arrived as a *second* assistant
    message with scaffolding wedged between, which fragmented a long answer
    across the history permanently.

    Only reasoning exists, because the round ended inside a `<think>` block.
    Re-issuing the original request makes the model reason from scratch and
    throws the work away: measured at 12,907 tokens of thinking discarded
    across three identical retries, 393 s for one turn. So the reasoning is
    fed back too. It still never reaches the stored history — `to_message()`
    omits thinking, and this request is rebuilt fresh every round.

    Nothing at all was produced — re-issue unchanged and let the loop's
    empty-turn handling take over. Scaffolding an empty assistant message here
    would break alternation, the very defect the truncation fix prevents.
    """
    if written.strip():
        return messages + [{"role": "assistant", "content": written},
                           {"role": "user", "content": CONTINUE}]
    if reasoned.strip():
        return messages + [{"role": "assistant", "content": reasoned},
                           {"role": "user", "content": CONTINUE_THINKING}]
    return messages


def _merge_rounds(final: AssistantTurn, text_parts: list[str],
                  think_parts: list[str], tool_schemas: list[dict],
                  config: dict) -> AssistantTurn:
    """Fuse the rounds into one turn, so nothing above ever sees the seam."""
    final.continuations = max(0, len(text_parts) - 1)
    final.text = "".join(text_parts)
    # Re-run text tool-call recovery across the seam.
    #
    # Small models write tool calls into the message body, and `stream` recovers
    # them per round — but a call split by the cap parses in neither half, while
    # the *merged* text holds it intact. Without this the call silently degrades
    # into prose: the model says it will run a command and nothing runs.
    # Harness-Bench found precisely this class ("incomplete tool recovery")
    # dominates harness failures.
    if len(text_parts) > 1 and not final.tool_calls:
        _recover(final, tool_schemas, config)
    final.thinking = "".join(think_parts)
    if len(text_parts) > 1 and not final.truncated:
        final.finish_reason = "stop"     # the seam is closed; it completed
    return final


def stream_complete(
    model: str,
    system: str,
    messages: list[dict],
    tool_schemas: list[dict],
    config: dict,
) -> Generator[TextChunk | ThinkChunk | AssistantTurn, None, None]:
    """Stream one COMPLETE assistant turn, continuing past the token cap.

    A generation cap is the harness's constraint, not the model's, so writing
    more than it should cost nothing. Previously a truncated reply was stubbed,
    a user message reading "continue, be more concise" was appended, and the
    remainder arrived as a *second* assistant message — so a long answer was
    permanently fragmented across the history by scaffolding, and after three
    rounds the turn simply ended with the work lost.

    Here continuation happens below the loop: the partial reply is fed back as
    an assistant turn to resume, and every piece is merged into a single
    AssistantTurn. Nothing above this layer ever sees the seam, the history
    holds one clean message, and the model is never told to write less.
    """
    text_parts: list[str] = []
    think_parts: list[str] = []
    final: AssistantTurn | None = None
    rounds = max(1, int(config.get("max_continuations", 8) or 8))

    # Deliberately no clock bound on the turn.
    #
    # There was one — 240 seconds, after which the turn took the landing path.
    # It was wrong on its own terms. A seconds limit makes the same model doing
    # the same task behave differently on different hardware: land early on a
    # slow GPU, finish on a fast one. That is the harness working against the
    # model, and it is the one absolute in a design whose rule is that the
    # context length is the only absolute anyone edits.
    #
    # The turn is still bounded, by two things that are relative to the model
    # rather than to the machine: `rounds` below, and the window check in
    # `_room_to_continue`. Both are real constraints — exceeding the window
    # would 400 the request. Impatience is not.
    started = _time.monotonic()

    for attempt in range(rounds + 1):
        if attempt:
            yield Continuing(attempt, rounds, _time.monotonic() - started)
        request = _resume_request(messages, "".join(text_parts),
                                  "".join(think_parts))
        turn: AssistantTurn | None = None
        circling = None
        watch = _Deliberation(config)
        partial_think: list[str] = []
        try:
            for event in stream(model, system, request, tool_schemas, config):
                if isinstance(event, ToolDraft):
                    yield event
                    continue
                if isinstance(event, (TextChunk, ThinkChunk)):
                    yield event
                    if isinstance(event, ThinkChunk):
                        partial_think.append(event.text)
                        if (circling := watch.feed(event.text)):
                            # Closing the generator drops the connection, which
                            # is the only way to stop a generation in flight.
                            break
                else:
                    turn = event
        except RuntimeError:
            # The server became unreachable partway through a long reply.
            # Losing every round written so far would penalise the model for a
            # network fault, so keep what exists and report it as truncated. On
            # the first round there is nothing to keep, so the error stands.
            if not final:
                raise
            final.finish_reason = "network"
            break
        if circling:
            # Not a truncation and not a penalty: every token of reasoning is
            # carried into the landing round, which asks for a decision rather
            # than for less thinking.
            yield StoppedCircling(circling)
            reasoned = "".join(think_parts) + "".join(partial_think)
            landed = yield from _conclude(model, system, messages, reasoned,
                                          tool_schemas, config)
            if landed is not None:
                text_parts.append(landed.text or "")
                think_parts.append(landed.thinking or "")
                final = landed
            elif final is None:
                final = AssistantTurn(text="", finish_reason="length",
                                      thinking=reasoned)
            break
        if turn is None:
            break

        text_parts.append(turn.text or "")
        think_parts.append(turn.thinking or "")
        final = turn

        # A tool call is actionable now; only prose needs continuing. A
        # truncated call has malformed arguments and is dropped upstream, which
        # a fresh attempt then re-emits intact.
        if not turn.truncated or turn.tool_calls:
            break

        # Continuing costs window: each round resends everything written so far.
        # The loop compacts *before* this call, not during it, so without a
        # bound here a long enough reply would grow the request past the window
        # and 400 mid-turn — turning a reply that was merely long into a failed
        # turn. Stop while the answer is still intact and let it be reported as
        # truncated instead.
        # Reasoning counts too: when the cap lands inside <think> it is the
        # reasoning that gets resent each round, so measuring only the visible
        # text would let a long chain of thought overflow the window unseen.
        if not _room_to_continue(
                messages, system,
                "".join(text_parts) + "".join(think_parts),
                tool_schemas, config):
            # Out of room. If all of it went into reasoning and none into an
            # answer, ending here wastes the whole turn: the loop retries and
            # pays the same cost again. Measured at 11,704 tokens of thinking
            # and no tool call, twice over.
            #
            # So spend one last round asking it to land — reasoning preserved,
            # nothing discarded. This is direction, not a length instruction:
            # the model is not told to think less, it is told the deliberating
            # is over and it is time to act.
            if not "".join(text_parts).strip() and "".join(think_parts).strip():
                landed = yield from _conclude(model, system, messages,
                                              "".join(think_parts), tool_schemas,
                                              config)
                if landed is not None:
                    text_parts.append(landed.text or "")
                    think_parts.append(landed.thinking or "")
                    final = landed
            break

    if final is not None:
        yield _merge_rounds(final, text_parts, think_parts, tool_schemas, config)


def _calibrate_from_usage(usage: dict, messages: list[dict], system: str,
                          tool_schemas: list[dict]) -> None:
    """Correct the token estimator against the server's own count.

    Compare like with like. The server counts the *whole* prompt: message text,
    the tool schemas, and the chat template's per-message markup. Measuring that
    against the message text alone attributes all the fixed overhead to content
    density and inflates the ratio — a short request measured 5.81x that way and
    1.26x once schemas were counted, pinning the correction at its 4.0 clamp and
    shrinking every budget fourfold.
    """
    actual = usage.get("prompt_tokens") or 0
    if not actual:
        return
    from .context import calibrate, raw_chars
    est = raw_chars(messages, system) // 4
    if tool_schemas:
        est += len(json.dumps(to_openai_tools(tool_schemas))) // 4
    calibrate(actual, est)


def _assemble_tool_calls(acc: dict[int, dict[str, str]]) -> list[dict]:
    """Indexed argument fragments -> tool calls, in the order they arrived."""
    out = []
    for idx in sorted(acc):
        slot = acc[idx]
        if not slot["name"]:
            continue
        try:
            args = json.loads(slot["args"]) if slot["args"].strip() else {}
        except json.JSONDecodeError:
            # Malformed arguments — usually a truncation mid-JSON. Mark it so
            # the loop can drop it rather than dispatching garbage.
            args = {"_raw": slot["args"]}
        out.append({"id": slot["id"] or f"call_{idx}",
                    "name": slot["name"], "input": args})
    return out


def stream(
    model: str,
    system: str,
    messages: list[dict],
    tool_schemas: list[dict],
    config: dict,
) -> Generator[TextChunk | AssistantTurn, None, None]:
    """Yield TextChunk(s) as they arrive, then exactly one AssistantTurn."""
    base_url, api_key, bare = resolve_endpoint(model, config)

    payload: dict[str, Any] = {
        "model": bare,
        # The system message is first and byte-stable across turns — that's what
        # makes the server's prefix cache hit.
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": True,
        # Ask for the token counts. Without this an OpenAI-compatible server
        # sends no usage block on a streamed response at all, so the estimator
        # has nothing to calibrate against and silently keeps its own bad guess
        # (context.estimate_tokens). Costs one small final SSE frame.
        "stream_options": {"include_usage": True},
    }
    # Sampling is sent only when someone chose it: the user, in config, or a
    # mechanical call such as the summariser, which sets temperature 0.0 for
    # itself. Otherwise the server's defaults apply, and for a local model
    # those are the model card's own (server.sampling_args).
    for key in _cfg.SAMPLING_KEYS:
        if config.get(key) not in (None, ""):
            payload[key] = config[key]
    if "reply_share" in config:
        # Sent only when a caller asks for it. The agent's own turns do not:
        # a generation cap is the harness's constraint, not the model's, and
        # every seam it creates costs a full resend of the turn so far. Set
        # explicitly by mechanical calls that are bounded by definition.
        payload["max_tokens"] = _cfg.budget(config, "reply_share")
    if config.get("frequency_penalty"):
        # Only for greedy mechanical calls — see summarise_span. Left unset for
        # the agent's own turns, where penalising a repeated token penalises
        # repeating an identifier, which is the whole job.
        payload["frequency_penalty"] = config["frequency_penalty"]
    if config.get("disable_thinking"):
        # For mechanical work — summarising what already happened — reasoning is
        # not just slow, it is a failure mode: measured on this model, the
        # default spent its entire 300-token budget inside <think> and returned
        # an empty answer, which would cost a full generation and produce
        # nothing. With thinking off the same request took 4s instead of 10s and
        # actually answered. Not for the agent's own turns, where the reasoning
        # is the point — with one exception, the landing round (_conclude),
        # which runs only after reasoning has already gone on too long.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if tool_schemas:
        payload["tools"] = to_openai_tools(tool_schemas)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with _Heartbeat(config, "the model's first token"):
        resp = _connect(f"{base_url}/chat/completions", headers, payload, config)

    turn = AssistantTurn()
    parts: list[str] = []
    think_parts: list[str] = []
    think = ThinkFilter()
    # tool_calls arrive as deltas keyed by index; accumulate the argument string.
    acc: dict[int, dict[str, str]] = {}

    beat = _Heartbeat(config, "tokens from the model")
    beat.__enter__()
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            beat.saw_data()
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                continue

            if evt.get("usage"):
                turn.usage = evt["usage"]
                # Ground truth for the estimator: the server has just counted
                # this exact prompt, so the character heuristic is corrected
                # against it rather than trusted (context.estimate_tokens).
                _calibrate_from_usage(evt["usage"], messages, system, tool_schemas)
            choices = evt.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                turn.finish_reason = choice["finish_reason"]

            delta = choice.get("delta") or {}
            # Some providers put reasoning in its own field instead of inline.
            if delta.get("reasoning_content"):
                think_parts.append(delta["reasoning_content"])
                yield ThinkChunk(delta["reasoning_content"])
            if delta.get("content"):
                visible, thought = think.feed(delta["content"])
                if thought:
                    think_parts.append(thought)
                    yield ThinkChunk(thought)
                if visible:
                    parts.append(visible)
                    yield TextChunk(visible)
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    before = len(slot["args"])
                    slot["args"] += fn["arguments"]
                    # Throttled: one event per ~400 characters written.
                    if before == 0 or len(slot["args"]) // 400 > before // 400:
                        yield ToolDraft(slot["name"], len(slot["args"]))
    except (requests.ConnectionError, requests.Timeout, requests.ChunkedEncodingError):
        # The connection dropped partway. Retrying is not safe — the user has
        # already seen whatever was streamed — so keep the partial turn and mark
        # it truncated. The loop's continuation path then does the right thing:
        # preserve alternation, ask the model to carry on from where it stopped.
        turn.finish_reason = "network"
    finally:
        beat.__exit__()

    tail_visible, tail_think = think.flush()
    if tail_visible:
        parts.append(tail_visible)
        yield TextChunk(tail_visible)
    if tail_think:
        think_parts.append(tail_think)

    turn.text = "".join(parts).strip()
    turn.thinking = "".join(think_parts)
    turn.tool_calls.extend(_assemble_tool_calls(acc))

    # Small models write tool calls as text instead of using the native field.
    if not turn.tool_calls:
        _recover(turn, tool_schemas, config)

    yield turn


def _recover(turn: AssistantTurn, tool_schemas: list[dict], config: dict) -> None:
    """Parse tool calls the model wrote into the message body. Never raises."""
    if not config.get("recover_text_tool_calls", True) or not turn.text or not tool_schemas:
        return
    try:
        valid = {s["name"] for s in tool_schemas}
        recovered, cleaned, unknown = toolcalls.recover_tool_calls(turn.text, valid)
    except Exception:
        return
    if recovered:
        turn.tool_calls = recovered
        turn.text = cleaned
    elif unknown:
        # Never silently drop something tool-shaped: either the model is
        # hallucinating a tool or it's using a format we don't parse.
        sys.stderr.write(
            f"\n\033[33m[warn] model emitted a call to unknown tool(s): "
            f"{', '.join(unknown)} — not dispatched\033[0m\n"
        )
