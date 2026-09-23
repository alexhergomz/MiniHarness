# Harness failure catalogue

Every bug found in this harness, and the rule each one produced. Written as a
spec because the individual fixes matter less than the patterns: nearly all of
these were **silent** — nothing errored, nothing looked wrong, and the damage
surfaced somewhere that did not point back at the cause.

Ordered by the principle each violates, not by discovery date. `DESIGN.md` holds
the reasoning behind the architecture; this file holds what actually went wrong.

---

## 0. The rules

Every entry below is an instance of one of these. They are ordered by what
they cost when broken, not by how obvious they look.

1. **A budget must be measured against what is actually sent**, not against the
   part that is convenient to measure. (§1)
2. **Never penalise the model for the harness's own limits.** Caps, windows and
   retries are our constraints; the model should lose nothing to them. (§2)
3. **An absence of output is never a completion.** Empty turns, truncated turns
   and dropped tool calls are failures to recover from, not answers. (§3)
4. **History must stay well-formed at every instant**, including after a crash,
   an interrupt, or a resume. (§4)
5. **Degrade audibly.** If something must be lost, say so in the transcript. A
   loss the model can see is recoverable; a silent one is not. (§5)

And one about method:

6. **A plausible causal story about a change you just made is not evidence.**
   Measure before attributing. (§7)

7. **Prefer deleting a mechanism to adding one.** Every mechanism is a place a
   bug can hide, and the ones that appear to work are the best hiding places.
   (§5.7, §7.10)

Rule 2 has a corollary that cost more than any other single defect here: **the
model's reasoning is output, and discarding it is the harness losing work the
model already did.** Preserve it for the whole assistant turn — through every
tool call, until the model speaks its final answer. (§3.3)

---

## 1. Budgets measured against the wrong thing

### 1.1 The history is not the request — *3 of 12 task failures*

Compaction budgeted the *messages* against the window: `compact_at × llama_ctx`
= 13,926 of 16,384. The request that actually goes out also carries the tool
schemas (~700 tokens), the per-turn focus map (819) and the reservation for the
reply about to be generated (4,096): **5,616 tokens of overhead never counted**.
A conversation compacted to just under budget produced a ~19,700-token request.

```
request (17056 tokens) exceeds the available context size (16384 tokens)
```

A 400 that kills the run mid-turn.

**Fix.** `compact_at` is a fraction of the room actually left for history:
`(window − schemas − map − reply) × compact_at`. One implementation in
`config.history_budget`; a test fails if any module hand-rolls it again.

**Why it hid.** Every quantity was already a share of the window, which made the
arithmetic *look* principled while it was quietly incomplete.

### 1.2 Three call sites, two wrong

The loop's mid-turn check, the REPL's between-turn check, and `/compact` each
computed the budget independently. `/compact` used a hardcoded `window // 4`,
which also violated the rule that every token quantity is a share of context
length.

**Rule.** Define each derived budget in exactly one function and call it from
every site. Here that is `config.history_budget(config, schemas)`, used by the
loop's mid-turn check, the REPL's between-turn check, `/compact`, and the
continuation bound. Add a test that greps the package for hand-rolled variants
(a regex for `window * compact_at` and `window // N`) and fails if any module
computes it independently.

### 1.3 Compaction could not reach its own budget — *silent overflow*

Pass 2 rescues every user turn; pass 3 elides only *tool* output. A conversation
carrying more user text than the window therefore returned **over budget in
silence**, and the request died with the 400 compaction exists to prevent.
Tightening the budget in §1.1 made this reachable.

**Fix.** Pass 4 shortens each instruction individually (head and tail kept);
pass 5, only when the arithmetic is genuinely impossible, drops the oldest
instructions and says how many. Invariant now tested across seven conversation
shapes: `compact()` never returns over budget, and the task statement always
survives.

**Two near-misses inside the fix itself**, both worth recording because the fix
for a silent-loss bug introduced two more silent losses:

- The first pass 4 elided the *middle* of the merged user message — which,
  because rescued turns are concatenated, deleted 22 of 30 instructions outright
  while claiming to drop none. *A fix that asserts a property must be tested for
  that exact property.*
- Pass 4 then shrank user messages **largest first**, and in an agent run the
  largest user message is almost always the task statement. The one message the
  entire run depends on was the first thing gutted, leaving the model working
  from a stub of its own instructions. Now the task statement is shortened last,
  whatever its size. *"Biggest first" is a good rule for reclaiming bytes and a
  terrible one for choosing what to lose.*

### 1.4 Three overflow bugs in one family — and the guard that should have existed first

The same failure shipped three separate times, each found only after a 400 killed
a benchmark run:

| where | request built | window |
|---|---|---|
| compaction not counting schemas/map/reply (§1.1) | 19,041 | 16,384 |
| the continuation resend, unbounded | up to 63,377 | 16,384 |
| the landing round of §2.3 | 51,397 | 16,384 |

The third is the most instructive: it was introduced *while fixing the second*,
and it is called **precisely when the window bound has already tripped** — so
history + reasoning was over budget by construction, and the round added to stop
a turn being wasted would have become the very failure it exists to prevent.

Auditing each construction site by hand had now failed three times. The guard is
therefore a property, not a review: every request `stream_complete` can build,
under adversarial sizes, is measured — messages plus system plus schemas plus
the reply reservation — and must fit. Verified to fail on all three bugs
independently when each fix is reverted.

It was *not* enough, and §1.4b is why: it measures with `estimate_tokens` on
both sides, so it proves the arithmetic and cannot see a bent ruler.

**Rule.** After the same class of bug appears a third time, write a property
test that enumerates every code path that can produce the artifact, rather than
auditing sites by hand. Here: drive `stream_complete` with a scripted model at
adversarial sizes, capture every request it builds, and assert
`messages + system + schemas + reply_reservation <= window` for each one. Verify
the test by reverting each of the three fixes in turn — it must fail three
times, reporting 18,316 / 63,377 / 51,397 tokens respectively.

### 1.5 The ruler was wrong — root cause of the whole family

After §1.1–§1.4, with the property test green, a run still died:

```
request (16401 tokens) exceeds the available context size (16384 tokens)
```

The arithmetic was right. `estimate_tokens` was not. "4 chars ≈ 1 token" is
close for prose and code and badly wrong for what an agent actually handles.
Measured against the server's own tokenizer:

| content | est (4 c/tok) | actual | error |
|---|---|---|---|
| prose | 450 | 401 | 0.89× |
| Python source | 1,500 | 1,500 | 1.00× |
| tool schemas | 701 | 747 | 1.07× |
| traceback | 900 | 1,140 | 1.27× |
| JSON tool output | 1,275 | 1,982 | **1.55×** |
| `ls -l` output | 1,601 | 3,766 | **2.35×** |
| hexdump | 2,174 | 7,723 | **3.55×** |

A 9,152-token history budget can therefore be 21,500 real tokens. Every budget
in the harness — compaction, continuation, the landing round, the tool-output
cap, the focus map — sits on this one function.

**No constant fixes it.** `len/2.5` still undercounts a hexdump by 2.2× while
wasting 1.8× on prose. But the server reports the true prompt length with every
response, so the correction is *measured*: `calibrate()` compares its count to
ours and scales the heuristic. Asymmetric on purpose — it jumps straight to any
underestimate and decays back slowly, because undercounting kills the run while
overcounting only compacts early. Verified live: a request of dense `ls` output
estimated at 5,460 tokens, actual 13,033 (2.39×), and after calibration
`estimate_tokens` returned 13,033 exactly.

**One more trap inside the fix.** An OpenAI-compatible server sends *no* usage
block on a streamed response unless `stream_options.include_usage` is set. The
first version of this shipped without it, so calibration would have silently
never engaged — a self-correcting mechanism that quietly never corrects, which
is worse than none, because it invites trust. A test now asserts the field is
sent.

**Rule.** When one function underpins every budget, validate it against an
external source of truth, not against itself. The §1.4 property test called
`estimate_tokens` on both sides of its assertion, so it stayed green while real
requests overflowed. The external truth here is the server's own
`usage.prompt_tokens`, which requires sending `stream_options:
{include_usage: true}` — without that field an OpenAI-compatible server returns
no usage block on a streamed response and the calibration silently never runs.

### 1.6 A fixed output cap larger than the window

Tool output was capped at 30,000 characters — more than an entire 4 k context.
Now `tool_output_share`.

### 1.7 `max_tokens=1500` against a 16 k window (measurement bug)

A benchmark harness set an absolute reply cap that manufactured three truncations
per task. Nearly reported as a harness defect.

**Rule.** Make context length the only absolute value in the config; express
every other budget as a fraction of it. Note this makes ratio bugs
scale-invariant: doubling the window 16k -> 32k changed compaction frequency by
zero, because `tool_output_share` stayed at 45% of the history budget either
way. When budgets thrash, fix the ratios between shares — adding window will
not help.

---

## 2. Penalising the model for the harness's limits

### 2.1 Generation caps fragmented and truncated replies

Exceeding `max_tokens` is *our* constraint. The old handling billed the model
three ways: it appended *"continue, but be more concise"*; it made the
continuation a **separate assistant message with a user message wedged between**,
permanently fragmenting a long answer across history; and it gave up after 3
rounds with the remainder lost.

**Fix.** `provider.stream_complete` continues *below* the loop, feeds the partial
reply back as an assistant turn with a neutral resume instruction, and merges
every piece into **one** `AssistantTurn`. History sees one clean message. Live:
a 90-token cap produced a 357-token reply, one message, ending on a complete
sentence.

**Bounds, neither a quality lever.** Round count (`max_continuations`), and
window space — each round resends what was written, and the loop compacts
*before* `stream_complete`, so an unbounded reply would grow past the window and
400 mid-turn, converting "the answer was long" into "the turn failed".

### 2.2 Two layers sharing one budget

The loop-level truncation retry reused `max_continuations`, which the provider
had already spent — silently doubling the cap and hiding which layer was working.
Now `max_truncation_resumes`, separate and smaller.

### 2.3 Reasoning discarded when the cap landed inside `<think>` — *393 s for one turn*

Thinking is not stored in history, by design. So when the cap fell inside a
`<think>` block there was no visible text to resume from, and the guard added
for that case **re-issued the original request unchanged** — making the model
reason from scratch and throwing away everything it had just done.

Measured on a reasoning task, one turn:

```
wall=393s  continuations=3  finish=length  tool_calls=1
text chars=0             thinking=12,907 tokens
```

Zero visible output, and 12,907 tokens of reasoning produced and discarded
across three identical retries before the fourth attempt finally emitted a tool
call. It got there — by doing the same work four times. This is the §2 rule
violated by the fix written to uphold it: the guard prevented an alternation
break and created a fourfold waste loop in its place.

**Fix.** Reasoning is work, so it is fed back as the assistant's own words with
an instruction to carry on rather than restart. It still never reaches stored
history (`to_message()` omits thinking, and the continuation request is rebuilt
each round), and it now counts toward the window bound — otherwise a long chain
of thought overflows the context unseen, since it is the reasoning that gets
resent every round.

**Rule.** Audit every branch that concludes "there is nothing to resume from".
Reasoning is not stored in history by design, so the truncation handler saw an
empty `text` field and re-issued the original request — discarding 12,907 tokens
of reasoning across three retries, 393 s for one turn. Feed the accumulated
`thinking` back as an assistant message and count it toward the continuation
window bound, since it is what gets resent each round.

**Then the overshoot.** Carrying the reasoning forward took the turn from 393 s
to 211 s and three rounds to one — but it now ended with **no tool call at
all**, where the wasteful version had at least produced one. Cheaper and less
useful: the window bound cut in before the model had turned any of its thinking
into an action, and the loop then retried and paid again.

The missing piece is that running out of room is not the same as being stuck. So
when the window fills with everything spent on reasoning and nothing on an
answer, one final round hands the reasoning back and says the deliberating is
over:

> "You have reached the limit of the space available for reasoning. Stop
> deliberating and act now: make your next tool call, or give your answer using
> what you have already worked out above."

That is **direction, not a length instruction** — the distinction that separates
this from the "be more concise" of §2.1. The model is never told to think less;
it is told when thinking time is up. Nothing is discarded, and the turn produces
something actionable instead of nothing.

| | wall | rounds | reasoning | finish | tool call |
|---|---|---|---|---|---|
| discard and restart | 393 s | 3 | 12,907 tok, discarded | `length` | yes |
| carry reasoning forward | 211 s | 1 | 11,704 tok, kept | `length` | **no** |
| carry + land | 272 s | 2 | 9,427 tok, kept | `stop` | yes |

**Rule.** Pair every bound that halts work with a final action that produces
something usable. Carrying reasoning forward alone took the turn from 393 s to
211 s but ended with no tool call at all — cheaper and less useful, because the
window bound cut in before any thinking became an action. Adding one last round
that returns the reasoning and says "stop deliberating, act now" restored the
tool call at 272 s. Distinguish this from a length instruction: telling the model
when its time is up is direction; telling it to write less is charging it for
your cap.

### 2.4 A network blip discarded the whole reply

If round 5 could not connect, `_connect`'s `RuntimeError` propagated and every
round already written was thrown away. Now the partial reply is kept and marked
truncated. On round one it still raises — swallowing that would turn an
unreachable server into a silent empty turn.

### 2.5 Truncation alternation

Stripping a malformed tool call can leave an assistant message with no text and
no calls. Popping it produces two consecutive user messages, and that alternation
violation makes models misbehave in ways that look like harness bugs: the local
Qwen spams unrelated tool calls; other models re-emit their previous output.
**Never pop — stub it.**

---

### 2.6 A turn with no time bound, and no way to see it

`max_continuations: 8` with a reply budget of 8,192 tokens allows 73,728 tokens
of generation in a single turn. Measured on this hardware at 46.7 tok/s that is
176 s per round and **26 minutes for one turn**, longer as context grows — and
the harness emitted nothing for the whole of it, because the "continued Nx"
notice is only reported once the turn ends. Two long-horizon runs were
investigated as hangs on the strength of that silence.

**Fix.** Continuation announces each round as it starts, so a long turn is
never silent. That is the whole fix.

**A wall-clock budget was added here and then removed.** `max_turn_seconds:
240` took the landing path when the turn ran long — no truncation, nothing
discarded, which is why it looked defensible. It was not. A limit in seconds
makes the same model on the same task behave differently on different
hardware: land early on a slow GPU, run to completion on a fast one. That is
the harness working against the model, and it is an absolute in a design whose
rule is that the context length is the only absolute anyone edits.

It never demonstrably fired, either. The `turn_cap` flag in the scenario
results that was read as evidence for it is `max_turns` — the tool-round
budget the *test* sets — and the two were reported as the same thing for
several runs.

The turn stays bounded by `max_continuations` and by the window check, both
relative to the model rather than to the machine. Exceeding the window would
400 the request, so that bound is a real constraint. Impatience is not.

**And a heartbeat, because none of the above was enough.** Every stall
investigation in this project failed the same way: a blocking call says nothing
until it returns, so a request that never returns leaves no evidence at all. A
thread now ticks while a request is outstanding, resetting whenever data
arrives — a healthy stream stays silent, a stuck one reports what it is stuck
on. Verified: 3 ticks when idle, 0 while streaming.

`request_timeout` was 600 s, retried four times: **40 minutes of silence** on a
request that stops responding. It is a socket *idle* bound, not a total budget,
and a healthy stream sends a token every few milliseconds — so 120 s costs a
long generation nothing.

### 2.7 A turn that never converges — *resolved; it was the model, and it is documented*

**Resolution first, because the investigation below took four wrong turns.**

The runaway is a known Qwen3.5 defect, documented by Qwen: the model has no
strong stop-thinking signal, so it "rephrases the same logic and revisits the
same conclusions" until something cuts it off. Qwen reports **17.4%** of its
LiveCodeBench outputs had truncated thinking, **84%** of those repeating above
30%. Their own issue tracker shows it occurring at their recommended sampling
parameters, closed as *not planned*.

What actually fixed it here, in order of contribution:

1. **Telling the agent a negative result is an answer.** The loop always began
   at the same place: an open-ended check with nothing to find. It fixed the
   seeded bug, verified it, then could not conclude that the rest of the code
   was clean — three identical greps, every module read, then circling. One
   line in the system prompt ended it. The summariser had been given this exact
   fix long before; the agent never had.
2. **A deliberation stop** that ends a turn whose reasoning has stopped
   progressing, and hands every token of it to the landing round.
3. **Removing the reply cap** (§2.8), which turned the seams and resends into
   one uninterrupted generation.

Measured after all three, six independent runs of the task that used to hang:

| | |
|---|---|
| verdicts | **6 PASS, 0 FAIL** |
| wall clock | 61, 65, 73, 74, 75, 311 s (median **74 s**) |
| loops | **1 of 6**, caught at 5,004 tokens and still PASS |

The one outlier is the run where the loop fired. It used to reach 45,101
tokens and never finish; it now costs 5,004 tokens and completes. The 1-in-6
rate is consistent with Qwen's documented 17.4%, though six runs cannot
establish that.

**Two things that did not work, recorded so they are not tried again.** A
`frequency_penalty` on the agent's turns is blind here — llama.cpp's penalty
window is `repeat_last_n: 64` and the loop's cycle measured ~166 tokens, so it
never holds two copies at once, while still taxing the ordinary repetition
prose and code require. The DRY sampler does see the cycle and cut verbatim
repetition twentyfold (distinct-line ratio 0.019 → 0.412) — and the model
simply paraphrased instead, which is precisely the behaviour Qwen documents.
Worse, at `dry_penalty_last_n: -1` it also penalises reproducing the prompt,
and `Edit` exists to reproduce the prompt: three edits drifted off the exact
text and the model then overwrote a 1,002-line module with a fragment.

The original investigation follows, because the wrong turns are the useful part.

### 2.7.1 A reply cap that scales with the window scales the wall clock

`reply_share: 0.25` is a share, so raising the window from 32k to 128k raised
the generation cap with it:

| | 32k | 128k |
|---|---|---|
| reply cap | 8,192 | **32,768** |
| one round at ~39 tok/s | ~3.5 min | **~14 min** |
| × `max_continuations: 8` | ~28 min | **~112 min** |

And the model fills it. Observed live on the server slot mid-run:

    processing True | decoded 29342 | remain 3426 | cap = 32768

One reply, 29,342 tokens and still going, no tool call — 23 minutes in which
the harness printed nothing, because it prints tool calls and this turn made
none. On hitting the cap it would have gone to a continuation round and
resent everything for another 32,768.

The shares rule is right for *memory* and is what fixed §3.8: stored tokens
cost window, and the window is what scaled. Generated tokens cost wall-clock
linearly, and nothing scaled to match. A cap the model reliably fills is not a
safety limit, it is a target.

**Unresolved, and the wrong fix is obvious.** A clock bound is what §2.6 just
removed, for good reason. The question to answer first is why a 4B model emits
32k tokens for one step.

**It is intermittent, not systematic — the first write-up of this section
overstated it.** A later run at the same window, with the full token stream
dumped and attributed to each step, produced the opposite picture:

| | the runaway turn | a 28-call run, same config |
|---|---|---|
| reasoning, one step | 29,342 tokens | max **759 bytes** |
| reasoning, whole run | — | **7,626 bytes** over 28 calls |
| visible prose sections | 0 | **0** |

Normal per-step deliberation is one to three sentences. So the cap is not a
target the model reliably fills; almost always it uses a fraction of a percent
of it. The 29,342-token reply was read off the server's own slot and is real,
but it is a rare excursion, and a rare excursion needs a different fix from a
systematic one — the shares rule is not, on this evidence, what causes it.

What that changes: the "×8 continuations = 112 min" row above is a worst case
that has been observed once, not a per-task cost. Do not redesign the budget
around it. The open question is narrower than it looked — what distinguishes
the one step that ran away from the 28 that did not — and it needs the runaway
caught again, with the stream dump already running, before a mechanism is
written for it.

**Second sighting, larger, and still intermittent.** On a fresh run it fired at
call #3, immediately after a bare `Read` of a 1002-line file:

    still writing — continuation 1/8, 1222s so far
    still writing — continuation 2/8, 3184s so far
    /slots: n_ctx 131072 | n_predict 32768 | temperature 0.3

The prompt reached **70,831 tokens**, of which ~55,000 were text the model had
generated in that one turn — 53 minutes, no tool call. The continuation
machinery worked exactly as designed and announced itself each round (§5.2);
what it was announcing was a turn that should never have got that long.

The stochastic part is confirmed rather than resolved: **two later runs made
the identical bare `Read` of the identical file at the same step and neither
ran away**, one of them finishing the whole task in 9 calls. Same prompt, same
model, `temperature: 0.3` — so the trigger is not the tool call that precedes
it.

**What it writes — and a wrong answer that stood here for a day.** This section
previously said the model reasons in *visible* prose, citing "221 bytes of
`<think>`, then it switched channels". **That was wrong, and the error was in
the instrument.** The stream dump wrote think-chunks and text-chunks to one file
with no channel label, and the 221 B counter belonged to the turn *before* the
runaway. The prose that was read off it was the reasoning itself.

Measured properly — the same state replayed against the server, counting
`reasoning_content` bytes against `content` bytes, six samples:

| think | visible text | outcome |
|---|---|---|
| 12,386 B | **0** | no tool call, hit the cap |
| 12,480 B | **0** | no tool call, hit the cap |
| 16,507 B | **0** | no tool call, hit the cap |
| 13,957 / 13,003 / 12,566 B | **0** | same |

**Six of six, not one byte of visible output.** The runaway is a turn spent
entirely inside `<think>` — the same shape as §3.4, not the essay this section
used to describe. The reasoning is coherent and on-task; there is simply no end
to it, and no tool call at the end.

That changes which machinery is implicated. With no visible text, the
continuation path in `stream_complete` takes its second branch — the §2.3 fix,
which resends the accumulated reasoning as the assistant's own words so the work
is not thrown away. Correct in isolation, but it means each round re-sends 12–16
KB of thinking *as content* and asks for more, which is how one turn reaches
55,000 tokens across rounds. The fix for §2.3 and the cost of §2.7 are the same
mechanism seen from two sides.

**Lesson about the instrument, not the model:** a dump that merges two channels
cannot answer a question about which channel was used. Label the stream at the
point of capture, or do not use it as evidence.

**And it converges.** On the run where this was captured, the turn hit the reply
cap, took one continuation round (239 s), then emitted the right `Edit` and the
task proceeded normally:

    still writing — continuation 1/8, 239s so far
    continued 1x past the reply cap
    #5 [t+278s] Edit tools.py  window = lines[offset:limit] -> ...

So §2.7 is a **cost** problem, not a correctness one, and the machinery already
built for it behaves: the model was never truncated, finished its thought, and
acted on it (§2.1, §2.3, §5.2 all doing their job at once).

That reframes the fix. Nothing in the prompt can shorten a `<think>` block —
the model is not writing prose it could be told to trim, and cutting the reply
short re-opens §2.1 and §2.3, which cost far more than the wait.

**What drives the volume — and three wrong answers before the right one.** The
history controls it, but not in any of the ways first suspected. Twelve arms,
`max_tokens: 2048`, no per-turn tail unless stated:

| condition | thinking |
|---|---|
| task-relevant source **absent** | 294, 318, 331, 348, 367 B |
| task-relevant source **present** | 2,565 … 8,059 B |

Complete separation: the largest "absent" measurement is 367 B, the smallest
"present" one is 2,565 B. The model thinks about ten times harder once it can
see the code the task is about — which is not a defect, it is the job.

Everything else that was proposed as the cause failed:

- **The per-turn tail** (working set + focus map) — not the cause. It is the
  arm where the model most reliably *acted* (3 of 3 tool calls); without it the
  model often produced nothing at all.
- **Context bulk** — two arms of near-identical size (11,591 vs 11,682 B)
  differed only by which file was quoted, and the difference tracked relevance,
  not size.
- **Reading the same file twice** (a bare `Read` returning a shape summary, then
  an overlapping ranged `Read`) — this was written up here as the leading
  suspect and as the leading *fix*. It does not survive. The arm it rested on
  measured **2,565, 5,840 and 8,059 B across three identical runs**, so the
  ~2.5× it appeared to add is inside the noise.

**Why the ratios were never measurable.** At `max_tokens: 2048` a full reply is
about 8 KB, and the high arms repeatedly landed there — 8,059 B *is* the cap.
The samples are **right-censored**: an arm that hits the cap has not produced
"8 KB of thinking", it has produced *at least* that. Comparing means between two
censored arms is meaningless, and no number of samples fixes it. This should
have been checked before the comparison was run, not after it produced a
plausible answer.

That censoring also explains the earlier observation that it "fills whatever cap
it is given" — 7.5 KB at 2,048 tokens, 12–16 KB at 4,096. True, and the reason
the original framing of this section ("a cap the model reliably fills is a
target, not a limit") holds — but about the *thinking* channel, which no prompt
instruction can reach.

**Consequence: there is no context-shaping fix.** Nothing about how the harness
assembles the history makes this better, because the trigger is the model having
the relevant code in front of it. What remains is not prevention but
measurement and visibility — whether a smaller cap reaches a tool call sooner
(unknown, cheap to test), and a heartbeat so a four-minute silence is legible
(§7.11).

### 2.8 The reply cap and the reply reservation — both removed

`reply_share` did two jobs, and neither survived being looked at.

**As a reservation** (subtracted in `history_budget`) it limited in both
directions: it capped the answer at the size set aside, and it shrank the
history by that much whether or not the answer needed it. The reply is the
work; the history is what should yield. Removing it at 32k took the history
budget from **18,917 to 25,880 tokens, +37%**, with no other change.

**As a generation cap** it bounded nothing in aggregate. With
`max_continuations: 8`, a 32,768-token cap at 128k permits ~295,000 generated
tokens in one turn — more than twice the window. All it did was cut a long
answer into seams, each seam costing a full resend of everything written so
far. That is the loop §2.7 measured: cap → truncation → continuation → resend.
And tightening it is worse, not safer — at a 655-token cap the model burned all
8 rounds and produced **no tool call at all in 2 of 3 turns**.

So `max_tokens` is no longer sent for the agent's turn. It survives only as an
explicit argument where a bound is part of the task: a compaction note longer
than the span it replaces reclaims nothing.

Same audit task, same 32k window, same seeded bug:

| | cap + no penalty | cap + penalty | **no cap + penalty** |
|---|---|---|---|
| tool calls | 44 | 44 | 45 |
| wall clock | 2,737 s | 1,079 s | **777 s** |
| compactions | 16 | 10 | **4** |
| compaction cost | 767 s (28%) | 289 s (27%) | **77 s (10%)** |
| continuation seams | 3 | 0 | **0** |
| verdict | PASS | PASS | PASS |

**What is attributable and what is not.** The drop from 10 compactions to 4 is
deterministic — a 37% larger history budget means the threshold is crossed less
often. That accounts for 212 s of the 302 s wall-clock difference, so most of
the gain is explained by a mechanism rather than by luck. The remainder is
within the run-to-run variance of §7.14 and is not claimed. Single runs; the
verdicts are the solid part.

**What did not happen, and was the risk.** Removing the cap could have turned a
turn chopped into 8 resent rounds into one unbounded generation. It did not:
zero seams, and the first edit landed at t+85 s against t+606 s for the capped
run that hit §2.7. The model stops when it is done, given the chance.

---

### 2.9 The harness chose the model's sampling for it

Every request carried `temperature: 0.3`. Nothing measured it; it was a
plausible-looking constant. Qwen's own card says 0.6 for coding in thinking
mode, with `top_k 20, min_p 0.0` — and llama-server, left alone, used its own
defaults of `top_k 40, min_p 0.05`. So the model ran on a mixture: the
harness's guess for one parameter, the server's generic defaults for the rest,
and its authors' recommendation for none of them.

Low temperature is exactly what feeds the repetition loops §2.7 had to contain.
The harness was working against the model and calling it a default.

The fix is not a better constant. The catalog now records what each family's
card documents (`models.FAMILY_SAMPLING`), llama-server is launched with those
values, and the harness sends no sampling at all unless the user sets it. A
family whose card is silent gets the server's defaults — not a guess presented
as a recommendation. Mechanical calls that need their own settings, such as the
summariser at temperature 0.0, still set them explicitly for themselves.

A server started by hand has to be given the same flags, or it falls back to
llama.cpp's generic defaults. `server.sampling_args(path)` prints them.

## 3. Treating absence as completion

### 3.1 Empty turn read as "done" — *task silently unfinished*

Qwen3.5-4B ran the tests, spent ~173 tokens inside `<think>`, then emitted no
visible text and no tool call. The loop treated that as completion. The run ended
after one tool call with the bug unfixed — and from outside it looked like a
clean exit.

| | tool calls | outcome |
|---|---|---|
| before | 1 | task not done, tests still failing |
| after | 4 | **2 passed**, correct fix |

**Rule.** Treat an assistant turn with no text and no tool calls as a failure
to recover from, not as completion. Stub the message to preserve user/assistant
alternation, append a prompt to act, and bound the retries (`max_empty_retries`,
default 2). Measured on a 4B model: without this the loop exited after 1 tool
call with the task unfixed; with it, 4 calls and the tests passed.

### 3.2 Tool calls lost at the continuation seam

Text tool-call recovery runs per round inside `stream()`. A small model writing a
call as JSON in the message body — the reason that recovery exists — produces a
call cut in half by the cap: neither half parses, and the call **silently
degrades into prose**. The model says it will run a command; nothing runs.

**Fix.** Re-scan the merged text across the seam. This is the "incomplete tool
recovery" class Harness-Bench found dominates harness failures.

### 3.3 Reasoning discarded every turn — *the largest defect found*

Two separate things were conflated, and getting the first one right caused the
second to be got wrong.

**Reasoning must not sit in `content`.** Emitted inline, it landed in history
verbatim: it cost context on every subsequent turn and fed the model its own
half-formed thinking as if it were a conclusion. `ThinkFilter` splits it out of
the stream, holding back partial tags across chunk boundaries.

**But splitting it out is not the same as throwing it away.** Having removed it
from `content`, nothing put it anywhere else, so each request contained only the
prose the model had typed *after* thinking. Measured on one turn: **4,841
characters of reasoning produced, 96 characters of prose kept.** Verified
directly — request N+1 contained nothing whatsoever from turn N's `<think>`.

Consequences, all of which had been mis-diagnosed as other things:

* A run showing 30 near-identical `Grep` calls was read as the model looping. It
  was not looping. Each turn was a **fresh start**: the plan that motivated the
  search had been deleted before the results came back, so the model re-derived
  a plan, re-issued a search, and lost it again.
* A model-written compaction note came out as a bare list of function names.
  That was not a weak summariser — a list of names was all the transcript still
  contained to summarise.
* The duplicate-call suppressor was built to answer the symptom. Suppressing the
  repeat leaves the model with no plan *and* no data.

**The fix is the documented one, and the model already supported it.** Anthropic's
guidance: *"when you return tool results, you must pass the thinking blocks from
the assistant message back to the API, complete and unmodified"*, because *"an
assistant turn doesn't complete until the model finishes its full response,
which may include multiple tool calls and results."* The unit of preservation is
the **turn**, not the message.

Measured against this server (llama.cpp + Qwen), three candidate encodings:

| encoding | accepted | model actually saw it |
|---|---|---|
| `reasoning_content` on an assistant message **with `tool_calls`** | yes | **yes** |
| `reasoning_content` on a completed assistant message | yes | no |
| inline `<think>…</think>` in `content` | yes | no |

The chat template renders reasoning only while the turn is still open. That is
the correct behaviour, and it matches the guidance exactly: carry it through the
tool calls, drop it once the turn closes. So `AssistantTurn.to_message()`
attaches `reasoning_content` **only when the message carries tool calls** —
complete and unmodified, never truncated to "the last few thoughts".

Measured cost, since "keep it all forever" is the obvious objection: **none.**
The template renders reasoning only while the turn is open, and drops it once
the model gives its final answer — a closed turn came back at an identical 91
prompt tokens with and without it. Within an open turn *every* step is rendered
(225 vs 173 tokens for three steps), and a plan stated in step 1 was still acted
on after two intervening tool calls. So the correct policy costs nothing beyond
the turn it serves, and needs no eviction logic of its own.

Three consequences for the rest of the harness:

* **The estimator must count it.** Preserving reasoning puts ~450 tokens per
  step back into the request, and the estimator did not see them: measured
  against the server, an open turn came out at **0.17x** — a 6x undercount, and
  §1's entire crash family starts exactly there. Counting open-turn reasoning
  brings it to 1.11-1.13x. Count it only where the template renders it; charging
  for finished turns would be safe but would waste real context.

* **Compaction fires mid-turn**, so it must not strip `reasoning_content` while
  shrinking tool output — that removes the plan at the moment it is needed.
  Regression-tested.
* **Do not truncate it to "the last N thoughts."** An earlier attempt injected
  the last two thoughts as a *user* message, which both mis-attributed the
  model's own words to the user and dropped the plan that motivated the work.

Test the *content*, not the plumbing: assert that a conclusion stated only in
turn N's reasoning is present in request N+1. Both encodings that fail above are
accepted by the server without error, so any test that only checks the request
was built and the call succeeded passes while the model sees nothing.

### 3.4 A cap spent entirely on thinking

If the whole cap goes inside `<think>`, there is no visible text to resume from.
Scaffolding an empty assistant message there breaks alternation, so the original
request is re-issued and §3.1 takes over.

---

### 3.5 Compaction: one mechanism, no markers

Elision — replacing tool output with `[output elided]` and keeping the shape —
was the original mechanism, and it was still running *inside* model compaction
long after it was supposed to be gone: `compact_with_model` called it twice, to
build a "skeleton" of the span it had just summarised and again as a backstop.

Measured breakdown of one compaction (15,112 → 3,402 tokens):

| part | tokens | share |
|---|---|---|
| verbatim tail | 3,000 | **88%** |
| elided skeleton | 154 | 4.5% |
| the note | 135 | 4% |

So the skeleton was never what consumed context, and removing it saves 4.5%.
What actually matters is that the *content* of the span is gone either way —
the only question is whether a note carries it forward or nothing does.

**What replaced it.** User instructions verbatim, the recent tail verbatim, and
everything older replaced by a note the model wrote. Nothing stands in for
content the model can no longer read.

**What it costs, stated honestly.** Without elision a single tool result larger
than the whole budget cannot be made smaller — truncating it would be elision
under another name. The guarantee is therefore: everything except the single
largest message fits. `tool_output_share` is what keeps results far below the
history budget in practice, so the floor is only reachable at budgets no real
session uses.

**Two bounds that must be shares, not counts.**

* The verbatim tail was a fixed `keep_recent=8`. That is a token quantity in
  disguise and it fails in both directions: those eight messages were 88% of
  everything kept, and a 64k window discarded exactly as much as a 16k one, so
  the extra room bought nothing. It now grows to fill a share of the budget.
* Compaction fires mid-turn, so it must not strip `reasoning_content` from the
  open turn while shrinking — that removes the plan at the moment it is needed.

**The summariser is a mechanical call.** It gets one attempt, not the usual
three retries: there is a fallback right behind it, so retrying an unreachable
model just spends the backoff before doing what it was always going to do — and
it is the agent's turn that pays. Measured, the test suite went from 14 s to
77 s on those retries alone.

---

### 3.6 A note is believed absolutely, so it must not guess

Two failures from the same place, both traced turn by turn.

**A required section with nothing to put in it is a request to invent.** The
template demanded "What is still wrong, and the exact error." A run that had
executed no command at all produced `Exact Error: KeyError: 'max_file_lines'` —
an error it could not have seen, for a line that did not exist. The system
prompt already forbade invention; that was not enough, because the *template*
still demanded a filled slot. Sections that can legitimately be empty now name
the words to write when they are, and the next run's note read "Nothing has
failed yet."

**There are two kinds of absence and only one is a fact.** The same note said
the `DEFAULTS` dictionary's "contents are not visible" — after the model had
already read the 150 lines containing them. That sentence outlived the evidence:
the model re-read `config.py` nine more times and never made the edit. But
banning absences outright was also wrong, and the next run repeatedly grepped
for a name that did not exist yet, four times, because nothing recorded that the
search had already come back empty.

    record    what a command established is not there
    never     what you have not got round to looking at

The first stops a search. The second is a standing instruction to look again.

---

### 3.7 Compaction erased the record of work already done

`working_set()` is the derived block telling the model what it has established —
what it changed, how commands ended, what it has already looked at. It was
rebuilt from `messages`. Compaction replaces a span of `messages` with a note.
The edits were in that span.

Measured directly, same fixture, before and after one compaction:

    before   Changed so far:
               - miniharness/config.py — replaced 'compact_to'
               - miniharness/tools.py  — replaced 'lines = text'
    after    None

So after every compaction the model was told nothing had changed. The note
covering the same span said `nothing changed yet` as well — two independent
sources agreeing it had done nothing, when it had edited two files.

The mechanism built to stop re-reading ("one file read nine times in forty
turns" is in its own docstring) was being defeated by the other mechanism
running on the same data. §5.7 again, third instance.

**Fix.** The working set now derives from the transcript *and* an append-only
ledger the loop writes at dispatch, holding `(id, name, args, result[:200])`.
It is the one record compaction cannot reach. Entries merge by call id, so a
call present in both counts once, and a resumed session whose ledger starts
empty still derives what it can from the messages it loaded.

**But it was not the binding constraint** — see §3.8, which is the more
important finding. This fix is correct on its own terms and stays; it was
treating a symptom.

### 3.8 The window was the binding constraint all along

At `llama_ctx: 32768` on a real task, compaction fired **after nearly every
tool call** — 8 in the first 25 calls, each reclaiming ~10k tokens — and the
model never escaped a re-read cycle:

    #18 Read tools.py offset=230 limit=100
    #19 Read tools.py offset=0   limit=50
    #20 Bash pytest
    #21 Read tools.py offset=230 limit=100      <- identical to #18
    #22 Read tools.py offset=200 limit=100
    #23 Read tools.py offset=0   limit=50       <- identical to #19
    #24 Bash pytest
    #25 Read tools.py offset=236 limit=100

Run to 40 calls, still cycling. The hysteresis gap (`compact_at: 0.85` →
`compact_to: 0.6`) is 25% of the history budget; one turn's output exceeded it,
so the gap bought nothing and every turn compacted.

Raising the server to `--ctx-size 131072` and `llama_ctx: 131072`, changing
nothing else:

| | 32k | 32k + ledger | 128k |
|---|---|---|---|
| first edit | ~2 min | t+59 s | **t+16 s** |
| first `pytest` | late | t+101 s | **t+69 s** |
| compactions in first ~160 s | several | 3 | **0** |

Zero compactions in 163 s and 18 calls, and the re-read cycle simply absent.

Every budget scaled with the one number, which is the shares rule working:
history budget 19,497 → 77,988, a single tool result 2,621 → 10,485 tokens —
enough to hold the whole region being edited instead of paging through it.

**The order of causes matters.** The amnesia in §3.7 was real and measured, and
fixing it helped. It was still a symptom of compacting once per turn. Before
building a mechanism to survive compaction, check whether compaction should be
running that often at all.

### 3.9 …and raising it made each compaction expensive — *the flip side of §3.8*

Same run, past the first 160 s. Compaction fires rarely now, but when it does it
dominates the clock:

| compaction | reclaimed | cost |
|---|---|---|
| after call #18 | 51,933 tok | 161 s |
| after call #28 | 46,215 tok | ~150 s |
| after call #33 | 60,263 tok | 152 s |

**~460 s of the 708 s elapsed — about 65% of wall clock.** All model generation
in the same run was ~1,900 tokens, roughly 49 s. The harness spent nine times
longer summarising the work as doing it.

This follows directly from the §3.8 fix and should have been predicted. The
window sets both how *often* compaction runs and how *much* it must read: raise
it 4× and you divide the frequency by ~4 while multiplying the span by ~4. The
total is not obviously better — it is the same work, re-shaped from frequent and
cheap into rare and expensive. §3.8 measured the frequency and stopped there.

I assumed the cost was prefill — that `summarise_span` sends its own
`SUMMARY_SYSTEM`, shares no prefix with the agent's turns, and so pays a cold
pass over the whole span. **That was wrong, and measuring it took one bench:** a
39k-token transcript, summarised both ways against a warmed server.

| | own `SUMMARY_SYSTEM` | agent's system prompt |
|---|---|---|
| prefill | 27,893 ms / 39,181 tok | 28,221 ms / 39,414 tok |
| generation | **2,048 tok, 101.7 s** | 171 tok, 8.2 s |

Prefill is identical. Swapping the system prompt to share the cache would have
saved nothing, and the note it produced was *worse* — it reported "nothing
changed yet" with an `Edit` and a failing `pytest` sitting in the transcript,
writing from the tool documentation in the system prompt instead of from the
transcript. That is the §3.6 failure the separate prompt was introduced to fix,
reproduced on demand. It stays.

The cost is **generation**, and the cause is the summariser's own greedy
decoding. It emitted one line about the fix **42 times in a row**, ran to the
cap, and returned a note that was 13% distinct lines. Same prompt, same call:

| | generated | time | stopped because |
|---|---|---|---|
| `temperature 0.0` (as shipped) | 2,048 tok | 101.4 s | hit the cap |
| `temperature 0.0` + `frequency_penalty 0.4` | **290 tok** | **14.1 s** | finished |
| `temperature 0.0` + `repeat_penalty 1.1` | 314 tok | 15.3 s | finished |
| `temperature 0.3` | 2,048 tok | 100.6 s | hit the cap |

**The obvious fix is the wrong one.** Raising the temperature does not stop it —
the model rambles for the full cap either way, the repetition merely stops being
verbatim, and the determinism that keeps a line number from being resampled is
gone for nothing. A frequency penalty leaves decoding greedy and ends the loop.
`frequency_penalty` is used over `repeat_penalty` because it is the portable
spelling; both worked. It is set small deliberately — a note must be free to
repeat `max_file_lines` as often as the facts require.

Expected compaction cost after the fix: ~28 s prefill + ~14 s generation ≈ 42 s,
against ~150 s. The *cap* made this expensive rather than dangerous: with
`reply_share: 0.06` at 128k the summariser may spend 7,864 tokens, and a looping
greedy decode will spend all of them.

Two smaller facts from the same bench, worth knowing before optimising a prefix:

- an identical resend costs **4 tokens, 56 ms** — the cache works;
- appending one 1-token user message to a cached prefix costs **6,122 tokens,
  5.5 s**, not the 4 tokens the prefix rule would suggest. Reuse is granular,
  not exact, so "it shares a prefix" is not the same as "it is free".

**Confirmed in a live run, not only on the bench.** Same seeded task, same 32k
window, same 44 tool calls, both PASS — the only difference is whether the
summariser gets the penalty:

| | penalty off | penalty on |
|---|---|---|
| compactions | 16 | 10 |
| note length, mean | 5,148 chars | **2,958 chars** |
| note length, max | 6,136 chars | 3,881 chars |
| **notes that ran to the cap** | **13 of 16** | **0 of 10** |
| cost per compaction, mean | 47.9 s | **28.9 s** |
| cost per compaction, median | 54.1 s | 30.9 s |
| total compaction cost | 767 s | 289 s |

The row that matters is the cap row. Without the penalty the summariser ran to
its ceiling on 13 of 16 compactions — `reply_share 0.06` × 32k ≈ 1,966 tokens,
which is what a ~6,000-character note is. It was not writing more; it was
failing to stop. With the penalty it never once reached the ceiling.

**What this run does *not* establish.** Wall clock was 2,737 s against 1,079 s,
and it is tempting to read that as the fix being 2.5× faster. It is not
supportable: the control happened to hit the §2.7 runaway three times (two
turns continued once, one twice) and the treated run not at all. Per §7.14 that
alone can swing a run several-fold. The compaction *count* (16 vs 10) is
confounded the same way — runaway prose is itself something that must then be
compacted. Only the per-compaction figures and the cap row measure the changed
component directly, so only those are claimed.

**The general point:** before optimising a call's prompt, split its wall clock
into prefill and generation and look at which one you are actually paying.

---

## 4. Malformed history

### 4.1 Ctrl-C between parallel tool calls

One tool message per call, so an interrupt leaves the first answered and the
second dangling — a guaranteed 400 next request. `repair_history` stubs the gaps.

### 4.2 Resume did not repair — *crash-only, so invisible in testing*

`repair_history` ran only in the REPL's `KeyboardInterrupt` handler. A crash,
`SIGKILL` or closed laptop left a dangling `tool_call` **on disk**; resuming sent
it verbatim and the 400 landed one turn later, where it reads like a server
fault. Both resume paths now repair before anything is sent.

### 4.3 Resume did not move the write cursor

`/resume` restored the messages but left `session_id` bound to the *new* session,
so every turn after a resume appended to a different file. The resumed transcript
stopped growing at the moment it was resumed. `session_id` now lives on `State` —
the only place a command handler can rebind.

### 4.4 Malformed tool calls left in history

A tool call whose arguments failed to parse must be dropped from the turn *and*
from history, or it is a 400 with no matching response.

---

## 5. Silent degradation

### 5.1 Governance decay

Only the *first* user message was protected from eviction, so a constraint given
mid-conversation ("when you finish, write this token to DONE.txt") was silently
removed and the run ended without it. Violation rates in the literature rise
0% → 30% → 59% as this compounds.

Nearly free to fix: on a representative conversation the user turns were 70
characters of 9,070 — **0.8% of context carrying all of the intent**. All user
turns are now rescued, adjacent ones merged to preserve alternation.

### 5.2 Continuation was invisible

Continuation ran up to 8 extra generations in a single turn and reported
nothing, so an 8× wall-time increase was indistinguishable from a slow model.
The loop now yields `Notice("continued Nx past the reply cap")`.

**Rule.** Any mechanism that can multiply cost must emit a countable event.
Applied here to continuation, compaction, duplicate suppression and retries —
each is a `Notice` the caller can count, which is also what made the compaction
thrashing in §1.4 visible at all.

### 5.3 Unavoidable loss must be announced

When instructions genuinely cannot fit, they are dropped *loudly*, with a count
and an instruction to restate anything still in force.

### 5.4 The harness told the model something the transcript contradicted

Duplicate suppression answers a repeated read with *"the result is already
above — use it, or do something different."* True while the result is above.
Compaction elides tool output, and the set of "already answered" calls was
maintained *beside* the transcript rather than derived from it — so the moment
compaction ran, the harness began refusing reads the model genuinely needed
while asserting the data was visible. The model cannot comply with an
instruction to use something that is not there: it tries variations and burns
turns.

Latent for as long as the estimator was undercounting, because compaction
rarely fired. Fixing the ruler (§1.5) made compaction honest and frequent, and
this became the dominant failure in the very next run — **two tasks that had
passed began hitting the turn cap**, with 5 and 12 suppressed duplicates:

| task | before | after |
|---|---|---|
| amuse-install | PASS, 34 calls, 323 s | fail, 40 calls, `hit_turn_cap`, 12 suppressed |
| analyze-access-logs | PASS, 7 calls, 66 s | fail, 40 calls, `hit_turn_cap`, 5 suppressed |

**Fix.** The set is derived from the transcript on each round
(`context.live_tool_signatures`), so it cannot disagree with it: a result that
was elided is no longer "already answered", and a mutating call still
invalidates everything before it.

**Rule.** Derive any claim you make to the model from the transcript at the
moment you make it; do not maintain it in a parallel data structure. The
"already answered" set was a Python set updated on dispatch, so when compaction
elided a result the set still listed it and the harness refused a read the model
needed while asserting the data was above. `context.live_tool_signatures` now
rebuilds the set from the messages each round — an O(n) scan over a list capped
by the context window.

**Second-order lesson.** Fixing the estimator did not create this bug, it
*revealed* it — a correctness fix elsewhere raised the rate of a code path that
was always wrong. Expect the run after a foundational fix to surface latent
bugs, and do not read that as the fix being wrong.

### 5.5 Two deterministic bugs that looked like the model hallucinating

A retention probe — compact a transcript, then ask questions whose answers were
in the removed content — produced confident, specific, wrong answers:

| question | truth | after elision | after model notes |
|---|---|---|---|
| cmake version | 3.22.1 | **"3.28.1"** | **"3.28.2"** |
| which line errored | CMakeLists.txt:4 | wrong file | **"Line 198"** |
| stp_simple links against | minisat | **"the simple library"** | minisat |

The obvious reading — "compaction causes hallucination, as the literature
warns" — was wrong, and the tell was that **elision cannot invent anything**.
It is deterministic string surgery. If a deterministic component appears to have
fabricated a value, the fault is upstream of it.

Two real bugs, both mine:

**The shrink heuristic selected by position.** Keeping head and tail of a long
output retained six lines of `Unpacking pkg0` and fourteen of
`Processing triggers 187`, and discarded both
`CMake Error at CMakeLists.txt:4 ... Could not find Minisat` and
`Setting up cmake (3.22.1)`. The stated rationale — "the tail is where the
failure is" — is false for build logs, where the error sits in the middle and
the tail is chatter. Lines are now ranked by information: output of this kind is
families of near-identical lines differing only in a number, each family is one
fact, so one member is kept and the rest counted. **9,095 -> 238 characters with
every fact preserved.** First *and* last of each family are kept, because
digit-normalisation makes `def f0()`..`def f99()` one shape and those are a
hundred facts, not one.

**The summariser wore the agent's system prompt.** Told it was a coding agent,
it wrote agent prose about *activity* — "Installed cmake via apt-get" — and
dropped the value. It now has its own prompt whose rules target exactly that:
record concrete values not categories, copy identifiers verbatim, and never
write what you did not see, because *an invented version number is worse than a
missing one — it will be believed*.

Result: elision 1/4 -> **4/4**, notes 2/4 -> **4/4**, no confabulation anywhere.

**And the ranking reversed.** With elision fixed, it matches model-written notes
on retention at **64% of the tokens and no generation cost** — so the summariser
buys nothing here, and `model_compaction` stays off for a measured reason rather
than a cautious one. Scope: every fact in this trajectory was a concrete value
on a distinct line, which is precisely what family-collapsing preserves. Notes
should still win where a fact needs synthesis across lines rather than
extraction from one.

**Rule.** When a deterministic component appears to have produced a wrong
value, inspect its output directly instead of theorising. Elision is string
surgery and cannot fabricate `3.28.1`, so the fabrication had to come from the
model, which meant the context it was given was missing the real value. Dumping
the compacted transcript verbatim showed both causes in under a minute:
20 lines of retained `Unpacking pkgN` noise, and a note reading "Installed
cmake" with no version. Neither was visible in the aggregate score.

### 5.6 Duplicate tool calls — *a symptom, and the machinery built for it was worse*

`Read` of the same file 8 times in one task; one run issued 30 near-identical
`Grep`s. The response was a suppressor: recognise a repeated read-only call and
answer it from the transcript instead of running it.

**It was the wrong response, three times over.**

1. **The repetition was not a caching problem.** It was §3.3 — the model's
   reasoning was discarded every turn, so each turn began without the plan that
   had motivated the search. Those 30 greps were 30 fresh starts. With reasoning
   preserved the same task took 3 calls in 19 s.
2. **The mechanism did not work.** `live_tool_signatures` stored the model's raw
   argument string; the loop looked up `json.dumps(params, sort_keys=True)`.
   They matched only when the model happened to emit its keys alphabetically.
   Then it turned out nothing consulted the result at all — the returned set was
   assigned and never read. It had been dead code for its whole life.
3. **It hid a real bug.** Serving a cached result masked a pagination defect:
   the final page carried no bookmark, so an identical call found no resume
   point and started the file again. Measured once the cache was removed, a
   `Read` walked 7,920 → 47,525 → 7,920 and looped. The cache had been papering
   over an infinite loop.

**Rule.** When the model repeats itself, find out what it is missing. A
suppressor answers the symptom, and a suppressor that refuses leaves the model
with neither the plan nor the data. Every call now runs; a repeat is the model
asking again, and the answer to that is the answer.

Two things that *are* worth having, and are not suppression: pagination
bookmarks that let a repeated call continue where it stopped, and a terminal
message when there is nothing left — both are information the model can act on.

---

### 5.7 Mechanisms hide each other's bugs

The pagination loop above existed the entire time and was invisible because a
cache sat in front of it. This is the general shape of everything in this
document: each mechanism was added to fix a symptom, each added surface, and
each new surface produced defects that the *next* mechanism concealed.

Counted over one session: removing elision and duplicate suppression — 1,171
lines — immediately exposed three real bugs that had been masked (the pagination
loop, a fallback that dropped messages while reporting `freed=0`, and a verbatim
tail pinned at 8 messages so a 64k window kept no more history than a 16k one).

**Rule.** Prefer deleting a mechanism to adding one. Every mechanism you keep is
a place a bug can hide, and the ones that "work" are the best hiding places.

---

### 5.8 The harness told the model what it would not tell the user

`_write` has reported gutting since §7.15: *"This removed 999 lines"*. It says
it to the **model**, after the write. The permission prompt — the thing a person
answers before it happens — printed the tool name and the path.

So the approval gate for the change that destroyed a 1,002-line module was
`Write mod.py` and a `[y]es / [n]o` prompt, carrying no more information than
the approval for a two-character typo fix. The information existed; it was
addressed to the wrong party and arrived after the fact.

`preview.py` renders the pending call instead: a unified diff for Write and
Edit, a line delta, and — last, next to the prompt, where a long diff cannot
scroll it off the top of the screen — the loud case:

    30 lines -> 1 (+1 -30)
    @@ -1,30 +1 @@
    …
    !! this removes 29 of 30 lines (96% of the file)

It also answers a question the prompt used to leave to the person: `Bash` shows
whether the jail is going to refuse the command anyway, and `Edit` says when
`old_string` will not match, or will match only through the whitespace-tolerant
fallback. Approving something that cannot run wastes the answer and the turn.

Nothing here re-implements a tool. The preview calls the same `_safe_target`
and `bash_outside_jail` the dispatcher will, and where it cannot know the
outcome it says so rather than guessing.

### 5.9 Every change checkpointed, in a repository that is not yours

Watched, and the reason §5.8 exists: the model destroyed a module, noticed, and
ran `git checkout` to recover — against the user's *real* repository. Wrong
repository, and it would have taken their uncommitted work with it.

`/undo` covers the last 50 Write/Edit calls, in memory. It does not cover a
Bash command that deleted a directory, and it does not survive the process
exiting. The generalisation is a checkpoint after every accepted change, and
the only question is where the commits go. Not into the user's history: their
branch must not move, their index must not be staged, and a project that is not
a git repository at all is the case with the most to lose, not the case to skip.

So the checkpoints live in a **shadow repository** — a git dir under
`~/.miniharness/checkpoints`, pointed at the working directory with
`--work-tree`. The user's `.git` is never opened. `/checkpoints` lists them and
`/rewind` restores one, after showing what the restore would undo, because it
deletes files created since.

The baseline is taken **before** the first mutating call rather than after it.
"The tree before the agent touched anything" is the state people actually want
back, and it is the one state that cannot be reconstructed once something has
been written.

One hazard, found by a test that put the store inside the working tree: the
store must be excluded from its own snapshots. `~/.miniharness` sits inside the
home directory, so an agent pointed at `$HOME` would checkpoint its own
checkpoints — and the rewind then fails partway through, having already deleted
the file it was restoring. A test that reproduced it end to end is worth more
here than the reasoning that should have caught it first.

## 6. Dependencies and environment

### 6.1 A dependency decided a benchmark outcome

A task container ran Python 3.12 with **no `pip`**, so `import requests` failed
and the harness scored zero on a task it never attempted.

**Fix.** Every HTTP call is POST-JSON, GET-file or stream-SSE; `urllib` does all
three. `net.py` (~160 lines) replaced it. The agent core now imports nothing
outside the stdlib. `rich`/`prompt_toolkit` remain but are REPL-only.

### 6.2 Two regressions from that swap

Characteristic risk of replacing a library with a shim — *the surface you think
is in use is smaller than the surface actually in use*:

- `data={"q": ...}`: `requests` form-encodes a dict, urllib demands bytes. Web
  search would have failed at runtime.
- `params={...}`: absent from the shim, so the call raised `TypeError` past a
  handler catching only `RequestException` — crash, not degrade.

**Fix.** A test extracts every kwarg at every call site and asserts the shim
supports it. Verified to fail when `params` is removed.

### 6.3 VRAM detection returned 0.5 GB (iGPU) instead of 6 GB

`nvidia-smi` was broken by an NVML version mismatch. Now probes all sources and
takes the max, with a ctypes `libcuda` fallback.

The mismatch is still present and is worth naming, because it is silent and
survives reboots of everything except the machine: the kernel module is
**595.71.05** (loaded at boot) and the userspace libraries are **595.84**
(upgraded 2026-07-31, no reboot since). `nvidia-smi` fails with
`Failed to initialize NVML: Driver/library version mismatch`, so *any* VRAM
probe that shells out to it gets nothing. CUDA itself is unaffected —
llama-server initialises and runs normally against the same mismatch — so the
failure is confined to monitoring, which is exactly the case where a harness is
most tempted to trust a single source.

`/sys/class/drm/card1/device/mem_info_vram_total` reports **0.5 GB** on this
machine: it is the iGPU. The dedicated GPU's size came from llama-server's own
startup line, which is the most reliable source available here because it is
the process that actually allocated the memory.

### 6.3.1 Context was 4× cheaper than assumed — measured, not estimated

The server had been running at `--ctx-size 32768` while the hardware allowed far
more, and that single number was the binding constraint on every long-horizon
run (§3.8). Measured at 131072 on an RTX 4050 (5,772 MiB):

    model buffer  = 2572.86 MiB
    KV cache      =  800.00 MiB   (131072 cells, 8 layers, turbo3 K + turbo3 V)
    compute       =  490.00 MiB
    offloaded 33/33 layers to GPU        ~1.9 GB free

800 MiB for 128k tokens, because only **8 layers carry a KV cache** on this
model. A per-layer estimate assuming all 33 would have said ~3.3 GB and ruled
out the window that fixed the runs. `n_ctx_train = 262144`, and 256k fits in the
remaining headroom.

The general point: the KV cost of a window is a property of the specific model's
architecture, not a formula. Boot it and read the number.

### 6.4 Documentation in the repository beat the code at every search

Scenario 3 asks what happens when the reply cap lands inside a `<think>` block.
The fixture copied the whole repository, including `HARNESS_SPEC.md` — which has
a section titled *"A cap spent entirely on thinking"*. The model spent 18 of 23
greps on regex variations of `cap.*spent.*entirely`, mining prose instead of
reading code, and scored PASS while measuring nothing.

Removing three design documents from the fixture:

| | with the docs | without |
|---|---|---|
| calls | 25 | **3** |
| time | 437 s | **22 s** |
| compactions | 11 | **0** |

It then answered correctly, naming `stream_complete` and the `CONTINUE_THINKING`
mechanism. The documents were not merely an answer key — they were *noise that
won*, because prose is written in the vocabulary of the question and code is
not, and because the large results forced eleven compactions.

**Rule.** Two of them. A fixture that ships the answer key measures nothing. And
in real use, a repository full of design prose measurably degrades grep-based
navigation — worth knowing before writing 800 lines of it.

Access control was a separate question and was sound: absolute paths, `../`
traversal, symlinks pointing out, and grep/glob/write outside the working
directory were all refused when probed. The leak was the fixture, not the jail.

---

### 6.5 A slow test in the agent's inner loop — *111 s to 90 min*

The agent is told to run the suite, so it runs it on nearly every turn. A
catalog check that made eleven Hugging Face requests was added to that suite;
against slow DNS each run cost up to 110 seconds, and one long-horizon scenario
went from 111 seconds to over ninety minutes without failing anything.

**Rule.** Whatever the agent runs in its loop is part of the agent's loop.
Anything slow, networked or non-deterministic in there multiplies by the turn
count. Put it behind an environment flag and run it in CI.

### 6.6 Relative paths ignored the working directory

6 tool calls / 25 s became 2 calls / 6 s once `_resolve` honoured `_cwd`.

### 6.7 `Glob **/*.py` missed root-level files

`fnmatch` needs a literal `/`; `**/` must be made optional.

### 6.8 Nothing ever told the model where it was — *45 turns to 9*

§6.6 made relative paths work. It did not make the model *write* them. The repo
map groups by directory:

    Repository map (30 files), most relevant first:
      tests/  test_core.py  test_commands.py  ...

The model read `tests/` as a root and asked for `/tests/test_core.py`, then
`/workspace/tests/test_core.py` — `/workspace` being a container convention it
brought from training. Its own reasoning names the source:

> "I need to find the test file. Let me use the correct path based on the
> repository map."
> "Let me use the absolute path from the repository map."

The jail refused every one. It ran `pwd`, then `ls -la`, saw the real path, and
went back to `/workspace` anyway — **12 of its first 17 calls** spent on a path
that could not resolve, and the run died at the turn cap having never reached
the bug.

The working directory appeared in no prompt. The system prompt omits it on
purpose (byte-stability, §5.1) and the map — the very thing it was reading
paths from — never named its own root. So the fix goes in the map, which is
rebuilt per turn anyway:

    Repository map (30 files), most relevant first.
    Working directory is /path/to/repo — paths below are relative to it, so
    read `tests/test_core.py`, never `/tests/test_core.py`:

Same task, same model, same seed bug: **45 turns without reaching the bug → 9
tool calls and 120 s to a green suite**, including the model grepping for the
same mistake elsewhere before declaring done.

Two lessons worth separating:

- **A refusal is not an instruction.** The jail's message already named the
  permitted root, and the model still repeated the refused call a dozen times.
  Telling it up front cost ~30 tokens a turn; telling it after the fact cost
  the run. Prevention, not recovery.
- **The temptation was to make the jail forgiving** — silently rebase an
  out-of-jail absolute path onto `_cwd` when that resolves inside. It would
  have saved the same calls. It was rejected: a path jail that quietly rewrites
  paths is no longer a path jail, and the honest fix was to supply the missing
  information instead.

---

## 7. Method — how to avoid measuring the wrong thing

Every item here is a mistake made *while investigating*, and each one made the
system look worse than it was.

### 7.1 Attributing a regression without evidence — then trusting the refutation too far

A task regressed from PASS/482 s to a 661 s timeout. Continuation had just landed
and the story was compelling: 8 rounds × a 4,096-token cap is up to 32 k tokens
per turn. An A/B on that exact task refuted it — **zero continuation rounds in
both arms**. The causes were §1.1 and an orphaned process competing for the GPU.

Then the opposite error, immediately after: that single-task A/B was treated as
clearing continuation generally. A *different* task turned out to trigger it
constantly (§2.3), and the wall-time cost was real after all — 393 s for one
turn. The A/B was sound; it was one task, chosen because it had regressed for an
unrelated reason, and it could not support the conclusion drawn from it.

**Rule.** An experiment that fails to reproduce an effect has shown it is absent
*in that condition*. Picking the condition because something else went wrong
there makes it close to worthless as a general clearance.

### 7.2 Reading a result as better than it was

Reporting that carrying reasoning forward turned "nothing" into "a tool call".
It had not: the raw line for the discarded-reasoning run read
`tool_calls=1` — it *had* produced a call, on the fourth attempt, after doing
the same work four times. The summary had been written from memory of what the
change was supposed to achieve rather than from the output.

Corrected, the result is still good and differently shaped: 31 % less wall time
for the same outcome without repeating work — and the intermediate version was
an **overshoot**, cheaper *and* less useful, because the window bound cut in
before any thinking had become an action.

**Rule.** Re-read the raw output immediately before quoting it. A number that
matches what the change was meant to do is the easiest one to misread.

### 7.3 Killing the wrapper shell, not the process — *five times*

`pkill -f longhorizon.py` matches every process whose command line contains that
string — including the shell that is running the `pkill` itself, and including
an unrelated command launched later in the same compound statement. Each time it
looked like the benchmark had crashed. The fifth occurrence killed a scenario
run that had been started in the same shell invocation, and the failure was
reported as an exit code with no output, which reads exactly like a hang.

**Rule.** Never `pkill -f` from a shell whose own command line contains the
pattern. Resolve to PIDs first and exclude the current shell:

    for pid in $(pgrep -f longhorizon.py); do [ "$pid" != "$$" ] && kill "$pid"; done

Prose written down and violated four more times is not a lesson; the fix is to
never type the unguarded form.

### 7.4 A forgotten background run

A stale benchmark from a discarded approach ran for hours, contending for the GPU
and corrupting every wall-time number taken in that window.

**Rule.** Before any timing measurement, assert that nothing else is running.

### 7.5 Guards that never fail are not guards

Two tests passed vacuously: a `find_module` import blocker (dead in 3.12), and a
source-scan for "concise" that matched a *comment*. Both now verified to fail
when the fix is removed. **Every regression test must be seen failing once.**

### 7.6 Broken measurement reported as a finding

- A ground-truth resolver compared `.h`→`.h` only, reading 26% recall for a
  resolver that was actually at 100%.
- `--timeout=20` passed to pytest without the plugin installed, so **every
  problem reported failure** regardless of outcome.
- A prefix-cache probe used `max_tokens=1`, which does not commit the slot cache
  — invalidating a "339× speedup" claim that had to be retracted.

**Rule.** When a measurement makes the system look bad, suspect the measurement
first. It is wrong more often than the system is.

### 7.7 Give a mechanical call no room to be inventive

Reasoning and sampling are defaults worth questioning per call, not per system.
Measured on the compaction summariser:

- **Thinking on**: spent its entire 300-token budget inside `<think>` and
  returned *empty content* — a whole generation for nothing, then a fallback.
  When it did answer, it invented a distinction between `ONLY_SIMPLE` and
  itself.
- **Thinking off**: 9.5x faster (67 s -> 7 s) and more accurate — correct line
  numbers, exact error text, no confabulation.
- **Temperature 0**: pending. Same argument: summarisation is extraction, and
  sampling a line number has no upside.

**Rule.** Ask what each call is *for*. Deliberation and sampling earn their cost
on the agent's own turns and nowhere else; on a call that only restates what
already happened they are a source of error, not of quality.

### 7.8 Benchmarks that flatter agents

BenchJack found SWE-bench, WebArena, OSWorld and GAIA all exploitable via
specification gaming. Prefer human-authored suites: Terminal-Bench (93
contributors, 229 tasks → 89 kept, ~3 reviewer-hours each, human-written
solutions) and Harness-Bench, which isolates the harness and reports a
23.8-point spread between best and worst on identical tasks — dominated by
"schema violations, incomplete tool recovery, weak evidence grounding rather
than pure reasoning failures". That is the same class of defect as almost
everything in §1–§5.

---

### 7.9 Four wrong causes for one symptom — *unresolved*

An intermittent stall: a long-horizon run stops making tool calls and sits for
40–90 minutes. It is **not diagnosed**, and it is recorded here as an open
problem rather than a solved one, because the sequence of wrong answers is the
useful part.

| proposed cause | how it was disproved |
|---|---|
| machine contention from other processes | reproduces on an idle machine |
| continuation rounds burning 26 minutes | zero rounds ever fired; `still writing` never once emitted |
| a pathological request payload | captured the payload and replayed it: **7.4 s** |
| the tool call before it | replayed against the same fixture: **0.00 s** |
| the 600 s socket idle timeout | reproduced with a 120 s bound in place — 66 minutes |

Every one of those was stated confidently before it was tested, and each was
argued from timing alone. What is known: the process sits in `do_poll` with the
connection open and CPU idle; it stalls at call #6, at call #12, and at call #0,
so it is not position-dependent; the server is healthy and its single slot is
free once the run dies. `faulthandler.dump_traceback_later` was armed and never
fired, which is itself unexplained.

**Rule.** A symptom with no evidence attached will attract one plausible cause
after another, and each will be wrong in a way that is expensive to discover.
Build the instrument first. The heartbeat in §2.6 exists because five rounds of
reasoning about this produced nothing, and it should have been the first move,
not the sixth.

**Corollary.** Do not present a fix as the cause of a symptom unless the link
was measured. Three of the changes above are correct and worth keeping on their
own terms — and none of them fixed this.

---

### 7.10 Deleting code finds bugs that reading it does not

Removing elision and duplicate suppression — 1,171 lines against 153 added —
immediately surfaced three defects that had been live the whole time and
invisible behind them (§5.7). A further pass over the dead surface found two
more: `base_url` raised `KeyError` on any partial config, and `jail_roots`
turned `extra_roots=[None]` into a root at `<cwd>/None`, silently *widening* a
path jail from malformed input.

Both sat in the 28 public functions with no test mention. That is where the
latent bugs were, and the way to find them was to remove the code around them.

### 7.11 A turn that calls no tool is invisible

A run went 23 minutes without printing a line and looked stalled. It was
generating a 32,768-token reply (§2.7). The harness logs tool calls, so a turn
that makes none logs nothing, and from outside the process there is no
difference between that and a hang.

Both non-invasive ways to look failed: `strace` cannot attach without root
(`ptrace_scope=1`, `Operation not permitted`), and the runner passed `None` for
the session, so there was no transcript on disk to tail.

`loop.run` already yields `TextChunk` and `ThinkChunk` — the runner was
discarding them. Writing them to a file behind an env var made the generation
visible immediately, and showed the split between reasoning and prose that
§2.7 turns on.

**The general form.** Instrument the thing you cannot see *before* reasoning
about it. Every stall investigation in this project failed the same way (§7.9,
§2.6): a blocking call says nothing until it returns. The heartbeat answered
"is it alive". This answered "what is it saying". Neither was available during
the investigations that guessed wrong.

### 7.12 Two statistics reported as the same thing

The scenario runner sets `turn_cap` from the notice `stopped after N tool
rounds` — that is `max_turns`, the tool-round budget the *test* chooses
(scenario 2 = 45, scenario 3 = 25, both exhausted). It was read and reported
for several runs as evidence that `max_turn_seconds` was firing. The time cap
never demonstrably fired in any run, and was removed on other grounds (§2.6).

A statistic named after one mechanism and computed from another is worse than
no statistic: it manufactured supporting evidence for a feature that was doing
nothing.

### 7.13 A recursive copy is not a sandbox — `.pyc` files carry absolute paths

The long-horizon scenarios run the agent against a *copy* of this repository,
with `HARNESS_SPEC.md` and `DESIGN.md` withheld: notes describing how the
harness fails are an answer sheet, not a task. The copy was made with
`copytree`, which took `__pycache__` with it.

A `.pyc` stores the absolute path of the file it was compiled from, and
`copy2` preserves mtimes, so the copied bytecode is still *valid* and Python
loads it in preference to recompiling. Every traceback in the sandbox therefore
named the original:

    co_filename = /home/<user>/Desktop/MiniHarness/tests/test_core.py

The agent read that path out of its own `pytest` output and spent **13 of its
first 14 calls** trying to open a file outside its working directory, in a loop
of identical `Read` and `Grep` calls, before trying a relative path.

Two things this establishes:

- **The path jail works.** Every one of those calls was refused —
  `refused to read … — outside the working directory (…)`. Nothing leaked. A
  sandbox that leaks its own identity through build artefacts would have handed
  the agent the withheld files by absolute path, and the jail is what stopped
  it. This is the first time the jail has been exercised by anything other than
  a unit test.
- **The isolation was mine to get wrong, and I got it wrong.** Withholding two
  markdown files is not isolation if the bytecode still points home. Copy source
  only — `ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")` — or build
  the sandbox from `git archive` and never inherit a build artefact at all.

**And the sandbox is still not sealed.** The jail covers `Read`/`Write`/`Edit`/
`Glob`/`Grep`; `Bash` is deliberately unconfined, because confining a shell
means parsing arbitrary commands. So an agent under test can reach the withheld
`HARNESS_SPEC.md` with `cat`, and nothing would stop it. Across every run here
it never tried — checked, not assumed — but *"it did not try"* is not isolation.
Any claim that a scenario is leak-proof is false unless the whole thing runs in
a container.

Worth noting for the harness itself: the refusal message names the permitted
roots, and the model still repeated the same refused call a dozen times before
adapting. A refusal that says where you *may* look is not the same as one the
model acts on.

### 7.14 One run of an agent task measures almost nothing

Same seeded bug, same task text, same model, same harness commit — three runs:

| window | calls | wall clock | compactions | verdict |
|---|---|---|---|---|
| 128k | 9 | 120 s | 0 | PASS |
| 32k | 13 | 429 s | 2 | PASS |
| 32k | 6 | 78 s | 0 | PASS |

**5.5× between the two runs at the same window**, and the difference is not the
harness: it is whether the §2.7 runaway happened to fire that time. When it
does, one turn costs 239 s and the prose it produces then forces the
compactions. When it does not, the task is six calls.

The consequence for measurement: a single run cannot support a claim about
wall clock, about how often compaction fires, or about whether a change helped.
The A/B that was supposed to prove the §3.9 fix in a live run failed for exactly
this reason — the control arm never compacted, so there was nothing to compare.
Forcing the condition (a task that must read the whole package) is the fix;
running the same stochastic task twice is not.

Verdicts are the exception: all three passed, and a pass/fail on a seeded bug
with a real test suite behind it is worth more than any of the timings.

**The same trap, one level down.** Sampling the model directly is not immune.
The §2.7 arms were three samples each, and one arm — re-run twice under
different labels in the same script — produced **2,565, 5,840 and 8,059 bytes**
of thinking on identical input. Every ratio computed from those means was noise,
including a 2.5× effect that had already been written into this document as the
leading cause and the leading fix.

Two rules from it:

- **Check for censoring before comparing.** The high arms were hitting
  `max_tokens`. A censored measurement is a lower bound, not a value; means of
  lower bounds cannot be divided. No sample count repairs this.
- **Separate groups, not ratios.** What survived was the one comparison with no
  overlap at all — five measurements at 294–367 B against seven at 2,565 B and
  above. A gap that a 3× variance cannot cross is worth reporting; a 1.3×
  difference between two noisy censored means is not.

### 7.15 Two patterns behind almost every fix that worked

Written after a day in which the harness went from failing its own audit task
to building a six-module ML toolkit from a spec at 12/13 on hidden tests.
Neither pattern is about the model.

**The harness knew something and did not say it.** Every one of these was found
by running real work, none by reading code:

| it knew | it said | now |
|---|---|---|
| a Write replaced 1,002 lines with 3 | "Updated (3 lines)" | what was removed |
| the suite went 1 failure → 3 | nothing | that it got worse |
| the working directory | nothing | named in the repo map |
| reasoning had stopped progressing | nothing | stops, keeps the work |
| one file rewritten 11× with 2 test runs all session | nothing | the count |

The repo-map case is the clearest: the model read `tests/` as a filesystem
root, asked for `/tests/…` and then `/workspace/…`, and burned 12 of 17 calls
on paths that could never resolve — while the harness held the answer and was
never asked. One line fixed it, and the task went from dying at the turn cap to
9 calls.

**Guard every path of a kind, not the one in front of you.** The deliberation
stop was written for the round loop. It then had to be extended to `_conclude`
— which is reached *because* the model is already circling, so the one call
left unbounded was the one guaranteed to be made in the worst state, and it ran
to 31,191 tokens over 31 minutes. Then to `summarise_span`, which goes straight
to `stream`: two notes ran to their 7,864-token ceiling at ~340 s each, 683 of
the 765 seconds a run spent compacting. Three discoveries of one omission,
each found only by a long-horizon run.

The same shape appeared in the jail: a static parser and a syscall watcher,
written separately, drifted until one refused `2>/dev/null` and the other did
not. Reconciling them introduced two regressions inside a minute — `/etc/shadow`
readable, and every nested path under `/tmp` — both caught by the tests written
when the jail shipped.

### 7.16 A guard defined by what it forbids will keep breaking

The Bash jail produced **six** false positives, every one found by running real
work rather than by reasoning about the rules:

| refused | actually |
|---|---|
| `mkdir …/site-packages/anyio/__pycache__` | importing a library |
| `openat /tmp/shXXXX` | a heredoc |
| `openat /tmp` | listing the temp root |
| `openat /etc/ssl/openssl.cnf`, `/dev/shm/sem.*` | importing numpy/sklearn |
| `python3 -c "print(a / b)"` | **division** |
| `execve ~/<a $PATH entry>/bin/python3` | `$PATH` search, and it failed |

Each was patched from the one symptom that surfaced, which is not a method. The
allowlist was eventually *derived* — 14 representative commands run under the
watcher with every denial collected — and that still missed the scientific
stack, because the workload was too narrow. Two rules generalised it properly:
`/etc` is readable wholesale rather than enumerated (three patches had gone
that way already), and a syscall that returns `-1` touched nothing.

The division case is the one to remember. A guard that refuses
`print(sum(xs) / len(xs))` gets switched off within the hour, and then protects
nothing at all.

### 7.17 A security test that runs its attack once measures almost nothing

Eighteen days after the jail was declared working, its escape test failed on a
run with the repository clean and no code changed. It passed on the next three.
The suite was not flaky in the usual sense — the jail was.

`watch_command` reads the strace log while the command runs, and it checked
whether the process had exited **after** its read rather than before. strace
buffers into that file, so the lines describing the last syscalls land as it
exits. A short command therefore finished, the watcher's read found a log
without them, the exit check fired, and the watcher returned reporting nothing.
Measured directly: **14 of 80 escapes succeeded** — `python3 -c` opened a file
outside the jail and its contents were returned to the model. After sampling
the exit flag before the read, so the final pass always covers a complete log:
0 of 300.

Two things about this are worth keeping.

The first is that the test was correct and still nearly useless. It attempted
each escape once, so it caught a one-in-six failure one run in four, and the
three green runs read as proof. The test now repeats twelve times and fails
deterministically without the fix. **Where the defect is a race, the number of
attempts is the test.**

The second is where it was found. Not by reading `watch_command` — it had been
read closely enough to earn six false-positive patches — but by running the
suite on a machine whose timing had drifted from the one it was written on.

### 7.18 Separate what defines the experiment from what runs in it

An evaluation has two halves and they need opposite treatment. The runner, the
project generators, the spec and the hidden grader **define** the experiment:
lose them and the numbers cannot be reproduced. The project the agent edits is
**consumed** by it: it is deleted and regenerated before every run, and nothing
is lost when it goes.

Both were got wrong, in opposite directions. Early runs pointed the agent at a
copy of the harness beside the real repository — the agent wrecked a file and
tried `git checkout` on the user's actual repo to recover, and any result from a
run like that is uninterpretable anyway, because the thing being measured and
the thing being damaged are the same. Then the tooling was left in the session
scratchpad, which put the durable half in the destructible place: `/tmp` cleanup
destroyed it **four times**, and on the fourth it took the only means of
reproducing a day of measurements.

It was recovered on 2026-09-04 by replaying every `Write`, `Edit` and patch
heredoc from the session transcript in order, and the recovery was *checked*
rather than assumed: the hidden ML grader was re-run against the reference
implementation and scored 13/13, the same as when it was first validated. A
grader is the one part of an evaluation whose faults are invisible — a broken
hidden test still yields a number — so `scenarios/validate_grader.py` exists to
be run before the number is believed.

The rule now holds by refusal, not by intention. Both generators begin by
deleting their working directory, so `workdir.disposable_or_die` rejects any
path inside the repository, under the user's home, or outside a temp root.

---

## 8. Things measured and rejected

Recorded so they are not re-attempted.

- **Progressive / NOTES.md compaction** — 2–4× slower and failed a run that
  eviction passed. Self-defeating: checkpointing costs turns, which costs
  context, which forces more eviction.
- **Symbol listing in the repo map** — redundant. 4/4 success with no map at
  all, because `FindSymbol` is grep.
- **PageRank over a symbol-name-joined graph** — the graph was false (710
  Python→JS edges). Name joins measure vocabulary similarity, not dependency;
  PageRank assumes citation semantics.
- **TriAttention** — ~19% throughput for unbounded quality risk on agentic
  workloads.
- **Re-injecting reasoning as a user message** — the model's last two thoughts
  appended to the request tail. It carried *something*, which is why it looked
  like it worked, but it attributed the model's own words to the user and threw
  away everything older than two turns, including the plan. The native encoding
  (§3.3) is both correct and free.
- **Inline `<think>` tags in assistant `content`** — accepted by the server
  without error and ignored entirely. This is the dangerous kind of wrong: a
  test that checks the request was built and the call returned 200 passes.
- **Elision, entirely** — deterministic and cheap, and it removes context
  outright. Removed, including from inside model compaction where it had
  survived as a "skeleton" (§3.5). Measured, the skeleton was 154 tokens of a
  3,402-token result; it was never what cost anything.
- **Duplicate suppression, entirely** — see §5.6. It answered a symptom of a
  different bug, it never worked (dead code), and it concealed an infinite
  pagination loop.
- **A `Write append` hint in the tool schema** — added because adding a test to
  a 1,700-line file had no affordance and cost 23 of 45 turns hunting for an
  `Edit` anchor. Used **zero** times across five runs: a schema description is
  not where a model looks. Moving the hint into the `Edit` failure message is
  the right shape, but it is *also* unproven — the failing runs never attempted
  an Edit at all; they explored and ran out of turns. Capability the model
  cannot find at the moment of need is not capability.

---

## 9. Checklist for a new harness

**Budgets**

1. Does every token budget count the schemas, the injected context, *and* the
   reply reservation — or only the part that was easy to measure? (§1.1)
2. Is each derived budget defined in exactly one place? (§1.2)
3. Can the shrinking mechanism always reach its budget, for every shape of
   conversation — or does it silently return over? (§1.3)
4. Is there a property test asserting **every** request that can be built fits
   the window, rather than a hand audit of the sites you thought of? (§1.4)

**The model's work**

5. Every cap: what does the model lose when it hits one? Including work that is
   not visible in the transcript, like reasoning. (§2.1, §2.3)
6. Every bound that stops work: is it paired with a way to *land* — or does it
   just convert a slow turn into a wasted one? (§2.3)
7. Every retry: does each layer have its own budget, or do two layers share a
   key and silently double it? (§2.2)
8. **Is the model's reasoning still in the request one turn later?** Not "is it
   captured", not "does the request build" — take a conclusion the model stated
   only in its thinking and assert it is present in the next request. Every
   encoding that silently fails is accepted by the server without error. Get
   this wrong and repeated searches, thin compaction notes and apparent
   forgetfulness all follow, and each looks like a different bug. (§3.3)
9. Does any instruction tell the model to produce *less*? Direction ("thinking
   time is up") is legitimate; length instructions are the harness billing the
   model for its own limits. (§2.1)

**Endings**

10. Every "the model is done" branch: could that be an empty turn, a truncated
   turn, or a tool call that failed to parse? (§3.1, §3.2)
11. Does tool-call recovery run on the *final* text, including across any seam
    the harness itself introduced? (§3.2)

**History**

12. Every mutation: is alternation still valid, and is every `tool_call`
    answered? (§4.1, §4.4)
13. Every crash path — not just the polite one: is what is on disk resumable?
    (§4.2)
14. After a resume, does the *write* cursor follow the read? (§4.3)

**Visibility**

15. Every eviction: would the user notice a constraint went missing? (§5.1)
14b. Any state kept beside the transcript: can it drift from what the
    transcript actually says? Derive it instead. (§5.4)
16. Every mechanism that can multiply cost: does it announce itself? (§5.2)
17. When loss is unavoidable, is it stated in the transcript? (§5.3)

**Mechanisms**

23. Does this new mechanism fix the cause or the symptom? If the model is
    repeating itself, re-reading, or forgetting, find what it is missing before
    building something to refuse it. (§5.6)
24. Could this mechanism be hiding a bug in the one behind it? Remove it and
    see what breaks — that is how the pagination loop, the silent `freed=0`
    fallback and the pinned tail were all found. (§5.7, §7.10)
25. Is guidance placed where the model actually looks — in the error it is
    reading — or in a schema description it will never read? (§8)

**Method**

18. Is the token estimator validated against the server's own count, or only
    against itself? (§1.5)
19. Before any timing measurement: is anything else running? (§7.3, §7.4)
20. Every regression test: has it been seen to fail? (§7.5)
21. Before quoting a number: has the raw output been re-read? (§7.2)
22. Before attributing a regression: does the experiment actually cover the
    condition being blamed? (§7.1)
23. When the same mistake happens a third time: what tool is missing? Prose
    written down twice and violated twice more is not a lesson. (§7.3)
