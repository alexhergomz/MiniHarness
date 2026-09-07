# MiniHarness — design

A local-first coding agent harness built on QOI philosophy: do a small number of
things excellently, delegate everything else to the environment.

> **Status: built and benchmarked.** 6,644 LOC across 16 modules, 5,565 LOC of
> tests, 385 collected. Two decisions locked in before implementation:
>
> - **Greenfield**, vendoring only `repomap.py` and `model_recommend.py`
>   (now `models.py`) from Promethean, which stays untouched as a reference.
> - **No llama.cpp fork.** MiniHarness builds an argv for stock `llama-server`
>   and nothing more. This is the constraint that keeps §5.4 honest.

Core principles, in priority order:

1. **Efficiency** — every token in the context window is paid for twice (prefill
   and attention). The harness's job is to spend them well.
2. **Local inference** — 8 GB VRAM is the design target, not an afterthought.
3. **Simplicity** — if a feature can't justify its schema tokens and its
   maintenance cost, it doesn't ship.
4. **Plug and play** — `miniharness` on a fresh box picks a model, downloads it,
   configures KV quant and context length, and starts serving.

---

## 1. What we're cutting, and why

Promethean is ~73 kLOC (~58 k excluding tests). Measured breakdown:

| Area | LOC | Verdict |
|---|---|---|
| `tests/` | 14,888 | Rewrite against the new surface |
| `modular/` (trading engines, video assembly, voice) | 8,833 | **Delete** |
| `demos/` (20+ demo-gif generators) | 8,769 | **Delete** |
| `commands/` (11 slash-command modules) | 5,848 | **Collapse to ~250** |
| `research/` (aggregator, ranker, citations, entities, classifier…) | 4,508 | **Collapse to ~150** |
| `web/` (browser terminal, server, api) | 3,510 | **Delete** |
| `tools/` (34 registered tools) | 2,927 | **Collapse to ~450** |
| `bridges/` (Slack, Telegram, WeChat) | 2,752 | **Delete** |
| `agent_tools/` (8 symbol-graph tools + repomap) | 2,221 | **Keep repomap, cut to ~550** |
| `multi_agent/`, `monitor/`, `plugin/`, `cc_mcp/`, `task/`, `skill/`, `checkpoint/` | ~5,400 | **Delete** |
| root (`promethean.py` 1760, `providers.py` 1195, `agent.py` 742, …) | 9,514 | **Rewrite to ~1,400** |

The bloat isn't mysterious. It's the standard failure mode: each feature was
individually defensible, none was ever removed, and a slash command plus a demo
generator plus a test module accreted around each one. A trading engine and a
WeChat bridge in a coding harness are the visible end state of "no removal
budget."

**Target: ~3,500 LOC core + ~1,200 LOC tests.** A ~16× reduction.

### The 34 → 9 tool cut

Currently registered: `Read Write Edit Bash Glob Grep WebFetch WebSearch Research
RepoMap FindSymbol GetCallers Outline Neighborhood PathBetween Imports
SearchFiles TaskCreate TaskUpdate TaskGet TaskList NotebookEdit GetDiagnostics
AskUserQuestion SleepTimer Think EnterPlanMode ExitPlanMode ReadEmail SendEmail
ReadPDF ReadImage ReadSpreadsheet WebBrowse`.

Keeping **nine**: `Read Write Edit Bash Glob Grep RepoMap FindSymbol WebFetch`.

> **What actually shipped: eight, and not those eight.** `WebSearch` came back.
> `RepoMap` and `FindSymbol` both stopped being tools — measured, not
> reconsidered. The map earns its place in the *system prompt*, ranked against
> the turn, where it costs no round-trip and no schema; a version with no map at
> all still scored 4/4, so a tool call to fetch one was never paying for itself.
> Symbol lookup folded into `Grep`: a grep that finds nothing falls back to a
> symbol search and answers with that, because the model reaches for grep first
> regardless. See §8 and HARNESS_SPEC §8. The reasoning below is kept as
> written — it is why the *shape* is right even where the count was wrong.

Rationale per cut:

- **`Think`, `TaskCreate/Update/Get/List`, `EnterPlanMode`/`ExitPlanMode`** —
  these ask the model to externalize reasoning into tool calls. A 9B model
  spends its limited capability filling out ceremony. Scratchpad reasoning
  belongs in the assistant turn, not in a tool round-trip.
- **`GetCallers`, `Outline`, `Neighborhood`, `PathBetween`, `Imports`,
  `SearchFiles`** — six tools that are all "navigate the symbol graph." Their
  combined schemas cost ~600 tokens on *every single turn*, forever. Folded into
  one `FindSymbol` that returns definition + callers + callees in one response.
  `PathBetween` (bidirectional BFS between two symbols) is a lovely piece of
  code that a model essentially never has a reason to call.
- **`ReadEmail`, `SendEmail`, `ReadSpreadsheet`, `WebBrowse`, `ReadPDF`,
  `ReadImage`** — not a coding agent's job, and mostly not a 9B model's job.
- **`GetDiagnostics`, `NotebookEdit`, `SleepTimer`, `AskUserQuestion`** — Bash
  covers the first two; the last two are loop-control concerns, not tools.
- **`Research`** — replaced by a bounded sub-loop, see §4.
- **`WebSearch`** — kept as an *internal* capability of the research sub-loop
  only, not a top-level tool. Coding turns shouldn't be tempted to search.

---

## 2. What the research says a great harness is

Two findings that actually change the design:

**The minimalist thesis (Pi / Mario Zechner):** a coding agent needs read, write,
edit, bash, and a system prompt under 1,000 tokens. Specialized tools like
"search the codebase" add tokens without adding capability, because frontier
models have been RL-trained on exactly this shape of task and already know what
bash is.

**The counter-consideration that matters for us:** that thesis is explicitly
premised on frontier models. Qwen3.5-9B has *not* been RL-trained up the wazoo.
It's inconsistent about tool-call wire format, it writes poor `rg` invocations,
and it doesn't reliably plan a multi-step exploration from a cold start. This is
the single most important design tension in the project:

> **Minimalism is a function of model capability. The right harness for a 9B
> model is smaller than Promethean and larger than Pi.**

Concretely, that's why `RepoMap` and `FindSymbol` survive the cut when Pi would
delete them. A frontier model can grep its way to understanding. A 9B model,
handed a ranked PageRank overview of the repo up front, performs dramatically
better than one told to go explore. That's not scaffolding for its own sake —
it's compensating for a specific, measurable capability gap. Everything else in
the graph toolkit goes.

**From harness-engineering practice (Osmani):** filesystem and git as durable
state; bash as the general-purpose escape hatch; hooks that enforce constraints;
compaction for context management; and the ratchet rule — *every line in your
system prompt should be traceable to a specific thing that went wrong.* We adopt
the ratchet rule as the governance mechanism that prevents re-bloating. No line
enters the system prompt without a failing transcript attached.

Also relevant: the hardest thing in harness design is long-horizon coherence —
early stopping, poor decomposition, incoherence across context windows. Notably,
scaffolding built to mitigate model anxiety became *obsolete* as models improved.
Design so that scaffolding is easy to delete later.

---

## 3. Architecture

Actual, as built:

```
miniharness/
  tools.py       the eight tools, deny-list, path jail, strace  1464
                 watcher, undo
  provider.py    OpenAI-compat streaming, think filter,          911
                 transparent continuation, deliberation stop
  context.py     stable prefix, focus map, value-ordered         735
                 eviction, model-written compaction
  __main__.py    CLI entry, REPL, slash commands, preview        559
  models.py      catalog, hardware detect, quant pick            508
  loop.py        agent loop, dispatch, permissions, recovery     505
  server.py      llama-server lifecycle, KV config               380
  research.py    the bounded research sub-loop                   256
  toolcalls.py   text tool-call parser                           253
  config.py      flat dict + TOML, context-relative shares       245
  repomap.py     git ls-files + relevance scoring, no deps       221
  checkpoint.py  shadow-repo snapshot after each change          201
  net.py         the stdlib HTTP client (see 5.3h)               171
  preview.py     what a pending tool call will do                145
  session.py     JSONL transcript, /resume                        83
  __init__.py                                                      7
                 total                                          6644
tests/
  test_core.py              units: safety, fit math, jail      2214
  test_loop_integration.py  loop vs. a scripted model           965
  test_provider_stream.py   SSE parsing, continuation           922
  test_scenarios.py         end-to-end tool sequences           497
  test_commands.py          slash commands                      318
  test_net.py               HTTP shim vs. a real socket         218
  test_resume.py            resume, incl. crashed sessions      164
  test_focus_map.py         per-turn map selection              163
  test_thinking.py          think-block filtering               104
                            385 tests, total                  5565
scenarios/
  longhorizon.py                                                258
  ml_project.py                                                 314
  scenario_project.py                                           423
  validate_grader.py                                             43
  workdir.py                                                     55
```



Sixteen modules against a ten-module budget: `toolcalls.py` stayed separate from
`provider.py` because 500 lines in one file to keep a file count round is the
wrong trade, and `research.py` is §4. `preview.py` and `checkpoint.py` are the
later two — both could have gone into `tools.py`, which is already the largest
file in the tree by 550 lines, and neither is about running a tool. No package-within-package, no plugin
registry, no MCP client, no dependency-injection layer. Readable in an afternoon.

### The three load-bearing pieces to carry over verbatim-ish

1. **`repomap.py`** — vendored from Aider. tree-sitter tags → identifier
   def/ref graph → personalized PageRank → token-budgeted rendering, cached in
   sqlite by mtime. This is the encoder-less indexing win, and it's genuinely
   excellent: no embedding model, no vector DB, no index server, ~160× speedup
   on repeat calls. Trim the aider-specific IO shims and the progress-spinner
   plumbing; keep the algorithm untouched.

2. **The truncation-alternation fix.** When `max_tokens` cuts a response
   mid-tool-call, the naive path pops the empty assistant turn, which breaks
   user/assistant alternation and makes Qwen spam tool calls. The fix is a
   `[output cut off at max_tokens]` stub that preserves alternation. Thirty
   lines that are the difference between a working local agent and a broken one.

3. **Text tool-call recovery.** Small models write tool calls as JSON or XML in
   the message body instead of the native `tool_calls` field. Parse and dispatch
   those. Non-negotiable for local models.

### Everything else in the loop

Permission flow: one deny-list regex for catastrophic bash (`rm -rf /`,
`dd of=/dev/sd*`, `curl | sh`, fork bombs), one sensitive-path jail (`~/.ssh`,
`~/.aws`, `~/.gnupg`, `/etc/shadow`), both firing even under `--accept-all`.
That's it — it's a guardrail against a confused model, not a sandbox, and it
should be honest about that. Container for untrusted models.

Slash commands: `/model /models /resume /compact /undo /config /help`. Seven,
in one file, ~250 lines.

---

## 4. Rabbit hole, radically simplified

Current: 1,131 LOC of workspace store + sub-question tree + BM25 synthesis +
contradiction detection + live event feed, plus 466 LOC of slash command, plus a
sandboxed background agent with slot save/restore.

Built: **one slash command, one markdown file, 217 LOC.**

```
/research <question>
```

A slash command rather than a tool, which resolves the tension in §1: the
capability stays, but its schema never costs tokens on ordinary coding turns,
and `WebSearch` stays out of the main tool set where it would only tempt the
model to search the web instead of reading the code in front of it.

Runs a bounded sub-loop (hard cap: N tool calls) with a three-tool
whitelist — `WebSearch`, `WebFetch`, `Read` (path-jailed to the workspace). It
appends every finding to `~/.miniharness/research/<slug>.md` as it goes:

```markdown
## <sub-question>
- <claim> — [source](url)
```

That's the whole data model. The file *is* the workspace, the *is* the resume
state, and the *is* the report. Want to resume? The sub-loop reads the file back
in. Want the synthesis? The file is already the synthesis — a markdown document
with claims and citations is what BM25-ranked reconstruction was trying to
produce anyway.

What's lost: contradiction detection, ranked retrieval across findings, the live
event feed. What's gained: the user can open the file in an editor mid-run, and
1,600 lines become 150. For a research feature on a local 9B model, that trade
is obviously correct.

---

## 5. Inference efficiency

Ordered by actual value per unit of complexity. This ordering is the whole point
— the biggest win costs zero lines of code and the fashionable one is a trap.

### 5.1 Prefix stability — the largest win, 0 LOC

llama.cpp caches the KV of a matching prompt prefix. If the first N tokens of
this turn are byte-identical to last turn, prefill is free. On an 8 GB card
prefilling 30 k tokens of context costs seconds; hitting the cache costs nothing.

This is a *discipline*, not a feature:

- System prompt, tool schemas, and repo map are emitted in a fixed order and
  never re-sorted (dict iteration order matters — pin it).
- No timestamps, no session IDs, no "current time is…" in the prefix.
- Repo map is refreshed only when the mtime fingerprint changes, and when it
  does, that's a deliberate, logged prefix invalidation.
- **Compaction rewrites the middle of the conversation, never the head.**

Most harnesses get this wrong by accident and eat a full reprefill every turn.
Getting it right is worth more than every trick below combined.

### 5.2 KV quantization — TurboQuant, minus the correction stage

TurboQuant (arXiv 2504.19874, ICLR 2026) is two stages: randomly rotate the
vector so coordinates become near-independent and Beta-distributed, apply an
MSE-optimal scalar quantizer per coordinate, then apply a **1-bit QJL transform
to the residual** to make the inner-product estimator unbiased. The paper reports
quality neutrality at 3.5 bits/channel and marginal degradation at 2.5.

The second stage is the source of the correction issues: it needs a second
buffer, a second kernel, and sign-bit bookkeeping in the FlashAttention path, and
it's where the fiddliness lives.

**Ship stage one only** — Walsh-Hadamard rotation + an optimal scalar quantizer,
no QJL. The rotation is the part that makes 4-bit scalar quantization of KV
actually work (it kills the outlier channels that wreck naive per-channel
quant); the QJL residual is a refinement on top.

**This is now the community consensus, not a compromise.** Every serious
llama.cpp implementation of TurboQuant independently arrived at the same
conclusion: the paper's Algorithm 1 (MSE-only, rotation + Lloyd-Max scalar
quant) outperforms the two-stage version in practice, and they all omit the QJL
residual correction. Reported: **turbo4 ≈ 3.8× vs f16 at +0.010 perplexity;
turbo3 ≈ 4.9× at ~1% PPL loss; under 10% throughput overhead.**

**Where that leaves MiniHarness.** TurboQuant is a fused-attention-kernel
change, so the harness cannot implement it — but it does not need to. The
`llama_kv_quant` config string is passed verbatim to
`--cache-type-k`/`--cache-type-v`, so:

```
/config llama_kv_quant=turbo3
```

...works the instant `llama_server_bin` points at a build that supports it. **No
code change, no fork maintained by us.** The no-fork decision means MiniHarness
does not *carry* the kernel; it never meant MiniHarness can't *use* it. That is
the whole point of keeping the inference contract at "build an argv."

Three routes to having it, in order of preference:

| Route | Status | Cost to us |
|---|---|---|
| Upstream llama.cpp | [Discussion #20969][tq-disc] active, integration spec drafted; **not merged** | zero — a default flip when it lands |
| A community fork's binary | Works today. CUDA, Metal, Vulkan, HIP/ROCm ports exist | zero — point `llama_server_bin` at it |
| Fork it ourselves | — | the thing we declined |

The default stays `q4_0` because it is the only one of these that works on a
stock build, and "plug and play on a fresh box" outranks peak efficiency.

**Two gates to check before assuming turbo3 works on the 8 GB target box.** The
published ports name gfx1201 (RDNA4), gfx1100 (RDNA3), and 7900XTX; the
development machine is an RX 6650 XT, which is **gfx1032 / RDNA2 and is not on
that list** — the Vulkan backend is the likeliest path but Vulkan coopmat
support on RDNA2 is limited. At least one implementation also requires
`head_dim == 128`, which needs confirming for the chosen model. Neither is a
blocker for the harness; both are blockers for the claim, so they get measured
rather than assumed.

The eval from earlier still stands and now has three arms: **q4_0 vs. turbo3 vs.
f16 KV**, on a fixed 200-task agentic benchmark scored on task success rate
rather than perplexity. Perplexity deltas of +0.010 tell you very little about
whether an agent still edits the right file on turn 40.

[tq-disc]: https://github.com/ggml-org/llama.cpp/discussions/20969

### 5.2b Why not PVQ

Pyramid Vector Quantization is a genuinely good suggestion on the merits, and it
is philosophically the most QOI option on the table: a fixed integer lattice on
the sphere, **search-free and codebook-free** — encode and decode in closed
form, nothing to store, nothing to train. [PVQ for LLMs][pvq] gets Llama-3 70B
to 3.25 bits/weight at 98% of downstream accuracy.

It still loses here, for two reasons that have nothing to do with the algorithm:

1. **It is primarily a *weight* quantization result.** The KV-cache vector-quant
   literature has moved via different methods (NSNQuant, FibQuant). There is no
   llama.cpp KV implementation of PVQ, so choosing it means being the first to
   write the kernel — the maximum possible fork burden, which is exactly the
   cost this project declined.
2. **The kernel shape is wrong for fused attention.** TurboQuant's rotation
   trick exists precisely to make *scalar* quantization near-optimal, and scalar
   dequant is branch-free and cheap inside a flash-attention inner loop. PVQ is a
   true vector quantizer; its combinatorial enumeration/indexing is branchy and
   much harder to fuse. That's an engineering argument specific to this setting,
   not a claim that PVQ is worse.

If PVQ ever gets a llama.cpp KV type, the same one-line config change picks it
up. That is the property worth protecting — not the choice of algorithm.

[pvq]: https://arxiv.org/abs/2410.16926

### 5.3 Prefix checkpointing — proposed, measured, and demoted

> **Default off — but for a different reason than first measured. Read on.**
>
> First measurement, on an RTX 4050 (6 GB) with Qwen3.5-4B Q4_K_M and `turbo3`
> KV, isolating prefill by erasing the slot between runs, said restoring was no
> faster than recomputing (2743→2709 ms at 4 k; 8152→8844 ms at 10.5 k). Worse,
> a repeated byte-identical prompt cost the same as a cold one, which would mean
> §5.1 — the largest claimed win in this document — was worthless.
>
> **That was the inference binary, not the design.** The same test against
> stock `llama-server` b10242, same model, same prefix:
>
> | | cold | warm (identical repeat) |
> |---|---|---|
> | wall clock | 58,651 ms | **173 ms** |
> | `prompt_n` (tokens evaluated) | 4200 | **4** |
>
> **339× on a repeated prefix**, with `prompt_n` correctly collapsing to 4.
> Prefix reuse is real and enormous. §5.1 is the single most valuable thing in
> this document — two orders of magnitude, for zero lines of code.
>
> ### Correction: the fork was *not* broken
>
> The first version of this section concluded that the TurboQuant+TriAttention
> fork breaks prompt-cache reuse, because `prompt_n` stayed pinned at the full
> prompt length on every request. **That conclusion was wrong, and the fault was
> in the probe.**
>
> Every one of those benchmarks used `max_tokens=1`. Some builds do not commit
> the prompt to the slot cache when only a single token is generated. Measured
> on the fork with an otherwise identical request:
>
> | probe | `prompt_n` over 3 identical requests |
> |---|---|
> | `max_tokens=1` | 1622, 1622, 1622 — no reuse |
> | `max_tokens=50` | 1622, **4, 4** — reuse works |
>
> So the fork's prefix cache is healthy for real generations. The "339× vs the
> fork" comparison was measuring a single-token probing artifact, not a property
> of the build. The stock-vs-nothing numbers above still stand (reuse is real);
> the stock-vs-fork *comparison* does not.
>
> `check_prefix_cache` now probes with `max_tokens=16` and a regression test
> pins it. The general lesson is worth more than the specific bug: **a
> measurement tool that is wrong in the pessimistic direction gets believed**,
> because it confirms the thing you were already suspicious of.
>
> Prefix *checkpointing* (the across-restart disk form) remains **untested on
> stock** and stays default-off — the in-memory cache already covers repeats
> within a server lifetime, which is the common case, and each checkpoint costs
> 79–121 MB.
>
> **The investigation also produced §5.3b, which is a real fix.**

The original reasoning, retained because the design logic still holds if the
reuse problem turns out to be fork-specific:

You asked about a hot/cold KV partition. The literature version — three-tier
GPU/CPU/NVMe offload, LMCache and llm-d style — is real and gets large TTFT wins,
but it's built for multi-tenant serving where many requests share cold prefixes.
Single-user, single-slot, on one 8 GB card, the multi-tier machinery has almost
nothing to bite on, and PCIe transfer of cold blocks on a consumer card is slow
enough to compete badly with just recomputing.

The version that *does* pay off locally, and that llama.cpp already implements:

**Serialize the KV of the stable prefix once, restore it on every new session.**
`llama-server` exposes slot save/restore endpoints. After the first prefill of
(system prompt + tool schemas + repo map) — call it 5–15 k tokens — save that
slot to disk keyed by a hash of the prefix bytes. Every subsequent session
restores it instead of reprefilling. Cold start goes from seconds to ~instant.

That's the hot/cold split that matters here: **hot** = the live conversation in
VRAM, **cold** = the frozen prefix on disk. ~80 LOC, uses an endpoint that
already exists, invalidates naturally via the prefix hash. It composes exactly
with §5.1 — the same discipline that makes prefix caching work makes the
checkpoint valid.

### 5.3b `--parallel 1` — the real find

Chasing the checkpoint failure turned up an actual bug, and this one is worth
more than the feature that led to it.

Left on its default, `llama-server` logs:

```
main: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true
```

Four slots, and requests land on whichever is free — the first agent turn went
to slot 3, a later one to slot 2. For a single-user coding agent that is
actively harmful:

- **It defeats §5.1.** The prefix KV a turn just built lives in one slot; the
  next turn may be served by a different one, so the carefully byte-stable
  prefix gets reprefilled anyway. The single biggest efficiency item in this
  document was being silently undone by a server default.
- **It made checkpointing a no-op.** `save` on slot 0 returned HTTP 200 with
  `"n_saved": 0, "n_written": 608` — a 608-byte header and nothing else,
  because slot 0 had never served a request. A green status code for a
  completely empty save is exactly the kind of failure that survives testing.

`server.py` now passes `--parallel 1` explicitly. One conversation, one slot,
deterministic reuse. After the fix, `save` reported `n_saved: 4082,
n_written: 78914528`.

### 5.3c Early stopping — the highest-impact fix found by testing

Not an efficiency item, but it came out of the same live testing and it matters
more than anything in §5.

Given a real multi-step task (run a failing test suite, find a cross-file bug,
fix it, re-run), Qwen3.5-4B ran the tests, spent ~173 tokens reasoning inside a
`<think>` block, then emitted **no visible text and no tool call**. The loop saw
"no tool calls" and treated it as completion. The run ended after one tool call
with the bug unfixed — and from the outside it looked like a clean, successful
exit.

An empty turn is never a legitimate completion: a model that is finished says so.
`_handle_empty_turn` now stubs the message (keeping alternation intact, same
discipline as the truncation fix) and asks the model to act, bounded by
`max_empty_retries`.

Same model, same task, same prompt:

| | tool calls | outcome |
|---|---|---|
| before | 1 | **task not done, tests still failing** |
| after | 4 (Glob → Read → Edit → Bash) | **2 passed**, correct fix |

The model was capable the whole time. The harness was giving up on it. This is
the clearest example in the project of the thing the research warned about —
*"a great harness makes a good model look excellent, a poor one makes an
excellent model look unreliable"* — and of why Promethean's "thinking-burnout"
handling, which this project deleted as bloat, was load-bearing. It is back,
and it is back the right way: because a transcript showed the failure.

### 5.3d What to optimise: tool calls, wall time, or context?

All three were measured, and they disagree — so the harness needs a stated
objective rather than an instinct.

Same task, same model, n=3 per arm, measuring everything at once:

| arm | tool calls (mean) | wall | **context consumed** | success |
|---|---|---|---|---|
| no map | 5.0 | 22.8 s | **5,510 tok** | 2/3 |
| map | 2.3 | 27.5 s | **8,752 tok** | 3/3 |

**Fewer tool calls consumed more context, not less.** The map's fixed ~2,100
token cost exceeds what it saves by avoiding two or three tool results. So
tool-call count is not a proxy for context, and (from §7) it is not a proxy for
latency either. Three plausible objectives, three different rankings.

The position this project takes:

1. **The objective is task success.** Nothing else is worth optimising directly.
   The literature is a useful warning here: failed agent trajectories are
   *consistently longer and more variable in step and token count* than
   successful ones ([LOCA-bench][loca]). Length is a **symptom** of failure, not
   its cause — so minimising trajectory length to make the numbers look like
   success is textbook Goodharting. The 2/3 vs 3/3 above is far too small to
   lean on, but it points the same way as the tail result in §7.

2. **Context is the binding constraint; wall time is not.** On a local box
   inference is already paid for — a slower turn costs nothing but patience,
   while exhausting the window is a hard failure that compaction can only
   partially undo. That is the specific reason this harness optimises differently
   from a metered cloud agent, where every token is billed.

3. **Wall time is a constraint, not an objective.** Keep it tolerable; do not
   trade reliability for it.

The practical consequence for the repo map: its context cost is **fixed and
paid once**, while the tool-result savings **accumulate per turn**. So it is
net-negative on context for short tasks and net-positive for long ones — the
opposite of the usual intuition that a big prefix hurts long sessions most.

[loca]: https://arxiv.org/pdf/2602.07962

### 5.3e Freezing the map, and admitting it is stale

The map is built once at session start and never rebuilt. That is deliberate:
re-rendering it would invalidate the KV cache for the whole prompt on every
write, and the cache pays off *per turn* while file changes are comparatively
rare. Systems that render components at session start and freeze them make the
same trade for the same reason.

But freezing creates the failure the same literature calls the top failure mode
of standing context files: **a stale map is worse than no map, because the model
trusts it completely.** And the agent makes the map stale *itself*, by editing
files during the session.

The fix is not fresher indexing — that is the expensive answer that loses the
cache. It is to carry the staleness in the read path so the model can discount
rather than act. `context.MAP_STALENESS_NOTE` prefixes the map with a plain
statement that it is a session-start snapshot which does not reflect later
edits, including the agent's own, and that files must be re-read before their
details are relied on.

The note is a **constant string** — deliberately no timestamp, no file count, no
commit hash. Anything varying would change the prefix every session and void the
very cache the freezing exists to protect. A test pins that it contains no
digits.

### 5.3f The focus map: dynamic, personalised, ephemeral

The static map was resident, generic and stale — three separate problems that
turned out to have one shared cause: **it lived in the system prompt.** That
placement is what forced it to be frozen (any change invalidates the whole
prefix), which forced it to be generic (it must serve every future turn), which
made it stale the moment the agent edited anything.

Moving it to the **tail of the request** unlocks all three at once. A changing
tail only invalidates from its own position onward, and everything after it is
new that turn anyway — so the per-turn cache cost is roughly the map's own size,
not the whole prompt. Freed from being frozen, it can be rebuilt every turn:

- **Fresh** — rebuilt against current mtimes, so it cannot be stale. The
  staleness disclaimer §5.3e added is deleted; there is nothing to disclaim.
- **Relevant** — PageRank personalised toward files already open (via the
  `FileTracker`) and identifiers in the request, so the budget lands on the task.
- **Ephemeral** — never written into history, so it costs its tokens once
  instead of accumulating.
- **Smaller** — 800-token budget instead of 2,048, because personalised tokens
  are worth more than generic ones.

And the system prompt is now **212 tokens and byte-identical across every repo
and session**, which is the strongest form of prefix stability available.

Measured, same task and model, ratios against the no-map arm *within the same
run* (absolutes drift between runs, ratios do not):

| | tool calls | wall | context |
|---|---|---|---|
| static map (§7) | 0.46× | 1.21× | 1.59× |
| **focus map** | **0.30×** | **0.85×** | **0.71×** |

The static map bought fewer tool calls by *spending* wall time and context. The
focus map improves **all three at once** — 1.0 tool calls vs 3.3, 9.8 s vs
11.5 s, 2,996 tokens vs 4,242 — at 3/3 success in both arms. n=3, so treat the
magnitudes as indicative; the sign flip on context and wall time is the point.

Two supporting fixes, both found by tests rather than intuition:

1. **Mentioning a symbol only helped if it was also referenced.** The 10× mention
   boost lived inside the defines-and-references intersection, so an entry point
   or a freshly written function — defined but not yet called — got no lift from
   being named. That is exactly the symbol people ask about. Files defining a
   mentioned identifier are now personalised directly.
2. **Users write prose, not identifiers.** "recover tool calls written as plain
   text" shares no exact token with `recover_tool_calls`, so exact matching only
   helped someone who already knew the symbol name — precisely the person who
   does not need a map. Mentions are now matched on *word tokens*
   (`recover_tool_calls` → {recover, tool, calls}), which connects "recover" to
   the symbol while *not* connecting "text" to `context`, as a substring match
   would. A token matching more than 5 % of the repo is discarded as
   uninformative — in a repo of widgets, "widget" selects everything and
   discriminates nothing.

### 5.3g The generation cap is the harness's problem, not the model's

A reply that runs past `max_tokens` is the harness hitting *its own* limit. The
old handling billed the model for it three separate ways:

1. It appended *"continue, but be more concise"* — telling the model to write
   less because we set the cap too low.
2. The continuation became a **separate assistant message with a user message
   wedged between it and the first half**. A single answer was permanently
   fragmented into three turns of history, and every later request paid to
   re-read the seam.
3. It gave up after 3 attempts, ending the turn with whatever had been written
   and losing the rest.

Continuation now happens in `provider.stream_complete`, *below* the loop. The
partial reply is fed back as an assistant turn with a neutral resume
instruction, and every piece is merged into **one** `AssistantTurn` before the
loop ever sees it. From `state.messages`' point of view the model simply wrote a
long reply; there is no scaffolding, no seam, and no instruction to write less.
A tool call ends continuation immediately — it is actionable now, and only prose
needs carrying on.

Two bounds exist, and neither is a quality lever:

- `max_continuations` (8) — so a model that never emits a stop token cannot
  loop forever.
- **Window space.** Each round resends everything written so far, and the loop
  compacts *before* `stream_complete` rather than during it. Without a bound, a
  reply long enough to matter would grow the request past the context window and
  400 mid-turn — converting "the answer was long" into "the turn failed", which
  is precisely the penalty this section exists to remove. Continuation stops
  while there is still room for one more reply under `compact_at`, in shares of
  the window like every other budget.

Hitting either is reported as truncation rather than silently swallowed.

**Measured, and it is insurance rather than a hot path.** A `amuse-install`
regression (PASS at 482 s → 661 s timeout) was initially attributed to
continuation multiplying wall time — 8 rounds × a 4,096-token cap is up to 32 k
tokens of generation per turn, which on a 4 B local model is tens of minutes.
An A/B on that exact task refuted it:

| arm | result | calls | wall | continuation rounds | truncations |
|---|---|---|---|---|---|
| `max_continuations=1` | fail | 40 | 348 s | 0 | 0 |
| `max_continuations=8` | PASS | 40 | 338 s | 0 | 0 |

Zero rounds in both, so the fail/PASS split is variance at nonzero temperature,
not an effect of the feature. The regression was §5.3j plus contention from an
orphaned benchmark process. The lesson is the ordinary one: a plausible causal
story about a change you just made is not evidence, and the instrument said so
immediately once it was asked.

The same table is the strongest argument that `reply_share` (0.25, or 4,096
tokens of a 16 k window) is over-provisioned — a reservation withheld from
history on every request that the model never once exhausted. Continuation is
what makes shrinking it safe, because exceeding the cap is now transparent
rather than lossy.

**The reasoning case, which turned out to be the expensive one.** On a thinking
model the cap frequently lands inside a `<think>` block: no visible text, but a
great deal of work. Thinking is deliberately absent from stored history, so the
first version of this code concluded there was "nothing to resume from" and
re-issued the original request — making the model reason from scratch. Measured
on a reasoning task, three rounds of that discarded 12,907 tokens of thinking
and took 393 s for a single turn — arriving, eventually, where one round of kept
reasoning would have.

Reasoning is work. It is now handed back as the assistant's own words, and it
counts toward the window bound, because when the cap lands in `<think>` it is
the reasoning that gets resent each round. That alone was still not enough: the
turn ended bounded, preserved, and *empty*, so the loop retried and paid the
cost again. Running out of room is not the same as being stuck, so a
final round ends the deliberation instead — restoring the action at a third
less wall time than the original:

| | wall | rounds | reasoning | finish | tool call |
|---|---|---|---|---|---|
| discard and restart | 393 s | 3 | 12,907 tok, discarded | `length` | yes |
| carry reasoning forward | 211 s | 1 | 11,704 tok, kept | `length` | **no** |
| carry + land | 272 s | 2 | 9,427 tok, kept | `stop` | yes |

The landing instruction says thinking time is up — never that the model should
think less. That distinction is the whole of §5.3g: *direction is not a length
instruction.* And the general rule: **a bound that stops work must be paired
with a way to land it**, or it converts a slow turn into a wasted one and the
retry pays twice.

If there is neither text nor reasoning, the request is re-issued unchanged —
scaffolding an empty assistant message would break alternation, and §5.3c's
empty-turn handling is the right owner of that case.

This is the general rule the harness follows: **work around the model, not
against it.** The same principle produced §5.3c (an empty turn is a failure to
recover from, not a completion) and the truncation-alternation fix (never pop an
empty assistant turn). In all three, the model was doing something reasonable
and the harness was punishing it for the harness's own constraint.

### 5.3h Two dependencies, and none in the agent core

Surfaced by Terminal-Bench rather than by design review. A task container ran
Python 3.12 but shipped **no `pip`**, so `import requests` failed and the
harness scored zero on a task it never got to attempt — a dependency deciding
the outcome of a benchmark that has nothing to do with dependencies.

Every HTTP call in the harness is one of three shapes: POST some JSON, GET a
file, stream SSE line by line. `urllib` does all three. `miniharness/net.py`
(~150 lines) mirrors the exact subset of the `requests` API in use — including
returning 4xx/5xx as a *response* rather than an exception, which `_connect`'s
retry logic depends on — so the five call sites changed by one import line each.

`rich` and `prompt_toolkit` remain, and should: rendering a terminal UI is
exactly the kind of thing QOI says to delegate. But they are REPL-only. The
agent core — loop, provider, tools, context — now imports nothing outside the
standard library, which is verified by a test that installs an import blocker
for third-party packages and imports the loop anyway.

`net.py` is tested against a real socket rather than a mock, because the
properties that matter are socket-level: that SSE lines arrive incrementally
(a buffering shim would still pass every mock-based test while destroying
streaming), and that a timeout is distinguishable from an unreachable host —
urllib hides the former inside `URLError`, and conflating them would send
retries down the wrong branch.

**Two regressions this introduced, both caught by reading the call sites rather
than by the suite**, and both worth recording because they are the characteristic
risk of swapping a library for a shim — the surface you *think* is in use is
smaller than the surface actually in use:

- `research.py` posts `data={"q": query}`. `requests` form-encodes a dict; urllib
  demands bytes. Web search would have failed at runtime.
- `server.py` passes `params={"action": ...}`, which `requests` turns into a
  query string. The shim had no such argument, so the call raised `TypeError` —
  past a handler catching only `RequestException`, so it would crash rather than
  degrade. Dormant today only because prefix checkpointing defaults off.

A test now extracts every kwarg passed at every `requests.get/post` call site in
the package and asserts it exists in the shim's signature, so the next one fails
at test time instead of at runtime. It was verified to fail when `params` is
removed — a guard that has never been seen to fail is not a guard.

Live coverage (opt-in, `MH_NET_TESTS=1`) confirms the paths a local socket
cannot: TLS, Hugging Face's 302 to a CDN, and `Range` → 206 for resumed
downloads. WebFetch, DuckDuckGo search and streaming completion were each
exercised end to end against the real endpoints.

### 5.3i Resume is where a broken transcript surfaces

Two defects, both found by writing the first tests for `/resume` rather than by
using it:

**History was only repaired on Ctrl-C.** `repair_history` ran in the REPL's
`KeyboardInterrupt` handler, so a session interrupted politely was clean, but a
crash, a `SIGKILL` or a closed laptop left a `tool_call` with no response *on
disk*. Resuming that transcript sent it verbatim, and the 400 landed one turn
later — where it reads like a server fault, not a resume fault. Both resume
paths now repair before anything is sent, and say how many calls they fixed.

**Resuming did not move the write cursor.** `/resume` restored the messages but
left `session_id` bound to the *new* session, so every turn after a resume was
appended to a different file. The resumed transcript stopped growing at the
moment it was resumed, and the work silently split in two. `session_id` now
lives on `State`, which is the only place that can be rebound from inside a
command handler.

The pattern is the same one as §5.3c and §5.3g: the failure is invisible from
the outside. Nothing errors, nothing looks wrong, and the damage only shows up
later somewhere that does not point back at the cause.

### 5.3j The history is not the request

The single worst defect found by benchmarking, and it had been silently costing
whole runs.

Compaction budgeted the *messages* against the window: `compact_at × llama_ctx`,
or 13,926 tokens of a 16,384-token window. But the request that actually goes out
also carries the tool schemas (~700 tokens), the per-turn focus map
(`repo_map_share`, 819), and a reservation for the reply the server is about to
generate (`reply_share`, 4,096) — 5,616 tokens of overhead that compaction cannot
shrink and was never counting. So a conversation compacted to *just under* budget
produced a request of ~19,700 against a 16,384 window, and the server answered:

```
request (17056 tokens) exceeds the available context size (16384 tokens)
```

That is a 400 that kills the run mid-turn, and it accounted for **three of twelve
task failures** in a benchmark — attributed at first to context pressure, which
is exactly backwards: the harness had room and threw it away by mis-measuring.

`compact_at` is now a fraction of the room actually left for history, not of the
raw window: history is budgeted at `(window − schemas − map − reply) × compact_at`,
9,152 tokens for a 16 k window, leaving 14,768 of 16,384 committed with margin.

Three call sites computed this budget independently and two were wrong — the
loop's mid-turn check, and `/compact`, which used a hardcoded `window // 4` that
ignored the shares rule as well. It now lives in exactly one place,
`config.history_budget`, and a test fails if any module hand-rolls it again. The
continuation bound in §5.3g uses the same function rather than a second copy.

The general lesson is worth more than the fix: **a budget must be measured
against what is actually sent, not against the part that is convenient to
measure.** Every quantity here was already a share of the window, which made the
arithmetic look principled while it was quietly incomplete.

### 5.3k Pending: deterministic decode for compaction

Summarisation is extraction, not generation. Sampling a line number, a flag
name or an error string is pure downside — there is nothing to be creative
about, and the failure it invites is exactly the one observed: the first live
summary invented a distinction between `ONLY_SIMPLE` and itself.

So the summary call should run at `temperature=0`, alongside the
`enable_thinking: false` it already sets. Both are the same argument — a
mechanical call should be given no room to be inventive:

| summary call | wall | output |
|---|---|---|
| default (thinking, temp 0.3) | 67 s | 130 tok, contained a confabulation |
| thinking off (temp 0.3) | **7 s** | 158 tok, precise, line numbers correct |
| thinking off, temp 0 | *pending* | expected: identical output run to run |

The third row is unmeasured and deliberately not applied yet: an A/B of
model-written notes against deterministic elision is in flight, and changing the
summariser between its two arms would confound the only comparison available.

A second reason to want it, beyond quality: run-to-run variance on these tasks
is currently larger than the effects being measured — `amuse-install` produced
`dup_suppressed=19, 324 s` and `dup_suppressed=0, 672 s` on *identical* code.
A deterministic summariser removes one source of that spread, which makes every
later comparison cheaper to run and easier to trust.

### 5.4 Things to deliberately not build

- **TriAttention** — dropping it, as you said. Worth recording why: the reported
  win is 28.4 vs 23.8 t/s at 4 K, ~19%, and eviction-based KV pruning trades
  quality for that in a way that's hard to bound on agentic workloads where an
  evicted token might be the file path you need in 40 turns. It also adds a
  fourth eviction-policy config surface (budget/window/sink/interval) to a
  project whose problem is too many knobs. 19% doesn't buy that.
- **Speculative decoding** — the recent public benchmark on Qwen3.6-35B-A3B
  across 19 configurations (ngram-cache, ngram-mod, and vocab-matched draft
  models) found *no* configuration achieving net speedup on a single RTX 3090.
  N-gram self-speculation only wins on highly repetitive or templated output.
  It's already a llama.cpp flag; expose it as a pass-through in
  `llama_server_args` and let anyone who wants it measure on their own box.
  Don't build support for it.

The general rule: an inference trick earns its place by being a flag we pass or
~80 lines we own. Anything requiring a maintained kernel fork does not survive
the simplicity constraint — which is precisely the lesson of maintaining two
llama.cpp forks that still haven't merged.

---

## 6. Plug and play

`model_recommend.py` is the other genuinely good piece of the existing codebase
and it survives mostly intact. The flow:

```
detect hardware (VRAM via rocm-smi/nvidia-smi, RAM via /proc/meminfo)
  → budget = VRAM - 0.8 GB runtime overhead
  → list GGUF quants from the HF repo tree, parse quant labels
  → for each: max_context = f(budget, weights_size, kv_gb_per_16k, kv_div=4)
  → rank by (quality × usable context), present a menu
  → download, write config, generate llama-server args, start
```

The context math already assumes `kv_div=4.0` for Q4 KV, so it stays correct
under §5.2. What gets cut: the alternate-pick machinery (`alt_quality`,
`alt_context`) that presents three variants per model. Show the recommendation
and one fallback; a menu with 3 options per model across 8 models is a menu
nobody reads.

First run, zero flags:

```
$ miniharness
No model configured. Detected: 8.0 GB VRAM (gfx1032), 16 GB RAM.

  1. Qwen3.5-9B      Q4_K_M   5.1 GB    →  180K ctx   (recommended)
  2. Qwen3.5-9B      Q5_K_M   6.2 GB    →   96K ctx
  3. Qwen3.5-4B      Q6_K     3.4 GB    →  256K ctx

Pick [1]: _
```

Then download → write `~/.miniharness/config.toml` → start `llama-server` with
`--jinja -fa -ctk q4_0 -ctv q4_0 -c 180000` → prefix checkpoint → REPL. No
build step required for the default path; the rotation patch is an opt-in
upgrade, not a prerequisite.

`--model ollama/qwen2.5-coder`, `--model gpt-4o`, etc. keep working — the
provider layer is OpenAI-compatible HTTP either way, and that's ~350 lines, not
1,195.

---

## 7. Migration order

1. `provider.py` + `loop.py` + `tools.py` (9 tools) — a working agent against a
   cloud endpoint. Validates the loop in isolation.
2. Port `repomap.py`, add `RepoMap`/`FindSymbol`. Measure the delta with and
   without the repo map — this is the load-bearing assumption of §2, so it
   should be measured, not assumed.

   **Measured (Qwen3.5-4B Q4_K_M, turbo3, 32 K ctx), 4-file project with a
   cross-file bug — fix the failing test, verify it passes:**

   | | tool calls | wall | path |
   |---|---|---|---|
   | map off | 4 | 25.2 s | **Glob** → Read → Edit → Bash |
   | map on | 4 | 26.0 s | Bash → Read → Edit → Bash |

   Both fixed it correctly. The map saved exactly one exploratory `Glob`. On a
   4-file tree a model can find things by guessing, so this says nothing about
   a large repo.

   **Large repo (295 files / 73 kLOC), locate code by concept, n=7 per arm,
   temperature 0.3:**

   | | tool calls med / mean / **max** | wall med / mean / **max** | success |
   |---|---|---|---|
   | map off | 8 / 7.7 / 14 | 37.3 s / 30.2 s / 53.5 s | 7/7 |
   | map on, alphabetical | **2** / 5.4 / **22** | **16.9 s** / 29.1 s / **74.5 s** | 7/7 |
   | map on, rank-ordered | 3 / **4.4** / **9** | 26.7 s / 32.4 s / **50.1 s** | 7/7 |

   The middle row looked like a win with a strange tail: much better median,
   *worse* worst case than having no map at all. A fatter tail from strictly
   more information is a contradiction, and chasing it found a real defect.

   **`to_tree` rendered the map alphabetically.** PageRank selected *which*
   symbols appeared but `sorted(tags)` then discarded the ordering — while the
   header told the model "most relevant first". On this repo that put
   `agent_tools/_shims.py`, a 106-line no-op shim, in first position and sank
   the 1,760-line entry point. Models read top-down and anchor on what they see
   first, so this is not cosmetic; it is the prompt lying about its own contents.

   Rank-ordering (third row) **cut worst-case tool calls 22 → 9 and worst-case
   wall time 74.5 s → 50.1 s**, and gives the best mean of any arm. It costs a
   slightly worse median (3 calls vs 2) — trading typical-case speed for
   predictability, which is the right trade when a 22-call run is what the user
   actually remembers. The change is justified on correctness grounds
   independently of the numbers.

   Even so, the honest summary is narrower than "the map is a big win": with
   n=7 and this much variance, the robust claims are that **every arm solved it
   every time**, and that **ordering controls the tail**. All three arms'
   means sit within a few calls of each other.

   **Why wall time barely moves even when tool calls nearly halve.** This looked
   contradictory, so it was measured directly (5 prompts, both conditions,
   temperature 0, reading llama.cpp's own `timings`):

   | | prefix | generated tokens/turn | generation time/turn |
   |---|---|---|---|
   | no map | ~212 tok | 91 (81–101) | 1.98 s |
   | with map | ~2,128 tok | 109 (71–133) | **2.63 s** |

   **The map does not just change how many turns happen, it changes what a turn
   costs.** Given more context the model reasons longer — ~20 % more generated
   tokens and ~33 % more generation time per turn. Fewer turns × more expensive
   turns ≈ the same wall clock.

   Prefill is *not* the culprit: the extra ~1,900 prefix tokens are prefilled
   once and then served from the prefix cache (§5.1), costing ~1 s across the
   whole run rather than per turn. Generation dominates — ~2 s of every ~2.6 s
   turn is token generation.

   Two consequences worth keeping:

   - **Tool-call count is not a proxy for latency.** Optimising the map to
     minimise tool calls optimises the wrong quantity. If wall time is the goal,
     the lever is generated tokens per turn — a terser system prompt, or
     capping reasoning — not a richer map.
   - The map's real payoff here is **fewer, more directed actions** (and a
     tighter tail once rank-ordered), not raw speed. That is still worth having:
     fewer tool calls means fewer chances to go wrong, which is what the tail
     numbers show.

   So §2 is **partly confirmed**: the map is a large win on the common case,
   which is what a user experiences, but it does not make the agent reliably
   faster and it costs ~2,100 prefix tokens every turn. "Dramatically better"
   was too strong; "much better usually, occasionally worse, never fatal" is
   what the data supports.

   The mechanism is worth recording because it is not the obvious one: **the
   answer was never in the map.** `tool_call_recovery` does not appear in the
   rendered map at all. What the map supplies is *orientation* — enough of the
   codebase's shape and naming conventions that the model writes a good `Grep`
   on the first try instead of guessing filenames. It is not a lookup table, and
   evaluating it as one would understate it.

   Method note: the first trial gave 11 → 2 calls and looked decisive. Trial 2
   gave 12 → 22 and reversed the sign. At this variance, n=1 is worthless and
   n=3 is barely better; the numbers above are the smallest honest sample.
3. `models.py` + `server.py`, local path end to end, prefix checkpointing.
4. `context.py` compaction (prefix-stable), `session.py`.
5. `research()` sub-loop.
6. Rotation patch + the 200-task KV eval from §5.2.

Steps 1–4 are the harness. If 5 and 6 never happen, it's still a better tool than
what exists now.

## 8. Keeping it small

The removal budget is the feature. Two rules:

- **The ratchet rule** — nothing enters the system prompt without a linked
  transcript showing the failure it prevents.
- **The schema tax** — every tool pays its schema token cost on every turn of
  every session forever. A tool that's called in under 1% of sessions is
  costing more than it returns. Review the call-frequency histogram quarterly
  and cut the tail.

---

## Sources

- [Agent Harness Engineering — Addy Osmani](https://addyosmani.com/blog/agent-harness-engineering/)
- [What I learned building an opinionated and minimal coding agent — Mario Zechner](https://mariozechner.at/posts/2025-11-30-pi-coding-agent/)
- [Pi Agent Harness (pi.dev): Minimal Coding Agent](https://explainx.ai/blog/pi-minimal-agent-harness-mario-zechner-guide-2026)
- [Building a Coding Agent From Scratch: Harness Architecture](https://www.decodingai.com/p/building-a-coding-agent-from-scratch-system-design)
- [awesome-harness-engineering](https://github.com/ai-boost/awesome-harness-engineering)
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (arXiv 2504.19874)](https://arxiv.org/abs/2504.19874)
- [TurboQuant reference implementation](https://github.com/OnlyTerp/turboquant)
- [KV cache offloading — LLM Inference Handbook](https://bentoml.com/llm/inference-optimization/kv-cache-offloading)
- [LMCache technical report](https://lmcache.ai/tech_report.pdf)
- [Native KV Cache Offloading to Any Filesystem with llm-d](https://llm-d.ai/blog/native-kv-cache-offloading-to-any-file-system-with-llm-d)
- [llama.cpp speculative decoding docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [Speculative decoding benchmark on RTX 3090 / Qwen3.6](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)
- [Aider repo map](https://aider.chat/docs/repomap.html)
