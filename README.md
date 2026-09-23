# MiniHarness

A coding agent that runs on hardware you already own. ~6,600 lines, 5,600 of tests.

Eight tools, one wire format, no fork of anything. See [DESIGN.md](DESIGN.md) for
why each of those is a deliberate number, and
[HARNESS_SPEC.md](HARNESS_SPEC.md) for every bug found building it and the rule
each one produced.

## Install

```bash
pip install -e .
```

Two dependencies, both for the REPL (`rich`, `prompt_toolkit`). The agent core —
loop, provider, tools, context — imports only the standard library, so it can be
dropped into any Python 3.10+ environment that has no package manager at all.

## Run

```bash
miniharness                          # first run picks and downloads a model
miniharness -m gpt-4o                # cloud (needs OPENAI_API_KEY)
miniharness -m ollama/qwen3-coder    # an existing local server
miniharness -p "fix the failing test in tests/test_auth.py"   # one-shot
```

First run with no model configured detects your VRAM, asks Hugging Face which
quantizations exist, and shows only the ones that actually fit — sized so the
model's **full context window** fits alongside the weights, not just the weights:

```
Detected: 8.0 GB VRAM (amdgpu)

  1. qwen3.5-9b       Q4_K_M       5.1 GB  ->   180K ctx (recommended)
  2. qwen3.5-9b       Q5_K_M       6.2 GB  ->    96K ctx
  3. qwen3.5-4b       Q6_K         3.4 GB  ->   256K ctx

Pick [1]:
```

Then it downloads, writes `~/.miniharness/config.toml`, and starts
`llama-server` with the right flags. No build step.

## Commands

```
/help    /model [name]    /models    /config [k=v]
/compact /think [on|off]  /undo      /resume [id]    /research <question>
/diff [-n]  /checkpoints  /rewind [-n|sha]  /clear
```

In a message, `@path` attaches a file (read through the same jail the model
uses, and counted as read so the first call can be the edit), and `!command`
runs a shell command yourself — its output goes along with your next message.
Tab completes `/commands` and `@paths`.

**What a turn looks like.** Each tool call is one line; a change is shown as a
line-numbered diff with the edited words marked, whether or not anything asked
first. A command's line is its verdict (`4 passed in 0.12s`), not its progress
bar. Anything slow — a command, a compaction — has a spinner rather than
silence. The turn ends with what it cost and what it changed:

```
● Edit(stats.py)
  ⎿  Updated stats.py with 1 addition and 1 removal
     4           total += x
     5 -     return total / (len(xs) - 1)
     5 +     return total / len(xs)
● Bash(python3 -m pytest -q)
  ⎿  1 passed in 0.01s
✓ 14s · 2 tool calls · 1 file changed · context 9% of 128k
  /diff to review · /rewind to undo
```

**Reasoning models.** Chain-of-thought (`<think>` blocks, or a `reasoning_content`
field) is **never stored in conversation history** — it would cost context on
every later turn and feed the model its own half-formed reasoning. It is
collapsed to `thinking…` on screen; `/think` expands the last one, `/think on`
streams it live.

## The tools

`Read` `Write` `Edit` `Bash` `Glob` `Grep` `WebFetch` `WebSearch`

Every tool pays its schema token cost on every turn of every session forever, so
the bar for a ninth one is high — and two that used to be here are gone. The
repo map is not a tool the model calls; it is built into the system prompt,
ranked against what the turn is about. Symbol lookup is not a tool either: a
`Grep` that finds nothing falls back to a symbol search and answers with that,
because the model was always going to reach for grep first anyway.

Write takes an `append` mode. Edit tolerates a whitespace-slipped `old_string`
when the match is unique, and otherwise answers *"line 412 is 87% similar — did
you mean this?"* rather than just refusing. Both exist because a 4B model was
watched losing whole turns to the alternative.

## What makes it fast locally

1. **Prefix stability.** The system prompt, tool schemas, and repo map are
   byte-identical across turns, so `llama-server`'s prefix cache hits and
   prefill is free. There are no timestamps in the prompt, tool order never
   changes, and compaction rewrites the *middle* of the conversation, never the
   head. This costs zero lines of code and is worth more than any decoding
   trick.
2. **One inference slot.** `llama-server` defaults to `n_parallel=4` and
   scatters consecutive turns across slots, so the prefix a turn just built
   isn't the one the next turn lands on — quietly undoing point 1. MiniHarness
   passes `--parallel 1`.

   (Prefix checkpointing — serializing that prefix KV to disk and restoring it
   on later sessions — is implemented but **default off**: it measured no
   faster than recomputing on an RTX 4050, and cost 79–121 MB per checkpoint.
   See [DESIGN.md §5.3](DESIGN.md).)
3. **Quantized KV cache.** ~4× less KV memory, which is what buys the long
   context on a small card. Defaults to `q4_0` because that works on a stock
   build. If you point `llama_server_bin` at a TurboQuant-capable llama.cpp
   build, `/config llama_kv_quant=turbo3` gets you WHT-rotated KV (~4.9×, ~1%
   PPL) with no code change — the harness only ever builds an argv, so it
   inherits kernel improvements for free. See [DESIGN.md §5.2](DESIGN.md).

Deliberately *not* built: TriAttention (~19% for an unbounded eviction-quality
risk) and speculative decoding (public benchmarks show no net speedup on
consumer GPUs for models this size — it's a pass-through flag if you want to
measure it yourself).

## Robustness

- **Transient failures retry** (connection drops, 429, 5xx) with backoff, but
  only before any token has been emitted — replaying mid-stream would duplicate
  output the user already saw. A mid-stream drop becomes a truncated turn and
  goes through the same continuation path as a `max_tokens` cut-off.
- **Interrupts leave valid history.** Ctrl-C between two parallel tool calls
  used to strand an unanswered `tool_call`, which 400s on the very next request.
- **Startup failures are diagnosed**, not dumped: an OOM says which knob to turn
  rather than pointing at 900 lines of tensor allocations.
- **The inference build is checked**, not trusted — see point 1 above.
- **You see the change before you approve it.** The permission prompt shows a
  diff for Write and Edit — and says loudly when a write removes most of a
  file — plus whether the jail is going to refuse a command anyway.
- **Every accepted change is checkpointed**, including the state before the
  first one. `/checkpoints` lists them, `/rewind` restores one. They live in a
  shadow git repository under `~/.miniharness`, so your own `.git` is never
  opened, your index is never staged and your branch never moves — and a
  project that is not a git repository at all gets them just the same.

## Tests

```bash
python -m pytest tests/ -q     # 341 tests
```

They cover the things that, when they broke, made the agent look fundamentally
broken: truncation alternation, tool-call recovery across four wire formats, the
bash deny-list, history validity across tool rounds and compaction, and the
memory fit math.

## Credits

The repo map is vendored from [Aider](https://aider.chat/docs/repomap.html)
(Apache-2.0). The tool-call recovery parser and the hardware/quant fit math come
from [Promethean](https://github.com/alexhergomz/Promethean), which this is a
deliberate 20× reduction of.
