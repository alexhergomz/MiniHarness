# Scenarios

Long-horizon runs against projects the agent is allowed to wreck.

## The rule

**The agent works on destructible things only.**

What defines an experiment — the runner, the two project generators, the spec
and the hidden grader — lives here, under version control. That is the backup.
What the agent touches is regenerated from this directory before every run,
lives under `/tmp`, and can be deleted at any moment without costing anything.

`workdir.disposable_or_die` makes that a rule rather than an intention. Both
generators begin by deleting their working directory, so it refuses to build
inside this repository, under your home, or anywhere outside a temp root.

This split was learned twice, in both directions. Early runs pointed the agent
at a copy of MiniHarness sitting beside the real repository; when it wrecked a
file it tried to `git checkout` the original to recover. Then the tooling
itself was left in `/tmp` — the durable half stored in the destructible place —
and `/tmp` cleanup destroyed it four times, taking the only means of
reproducing the numbers with it. It was recovered from the session transcript
on 2026-09-04 and put here.

## Running one

    cd scenarios
    MH_CTX=131072 MH_TASK=ml MH_TURNS=140 MH_WORK=/tmp/mh-ml-build \
        python3 -u longhorizon.py

| variable | meaning | default |
|---|---|---|
| `MH_TASK` | `fix` \| `audit` \| `refactor` \| `ml` | `fix` |
| `MH_CTX` | context window the harness budgets for | 131072 |
| `MH_TURNS` | tool-round cap | 45 |
| `MH_WORK` | where to build it — must be disposable | `/tmp/mh-scenario-csvstats` |
| `MH_PORT` | llama-server port | 8890 |
| `MH_DRY`, `MH_DRY_LAST` | DRY sampler, unset = off | off |
| `MH_NOPENALTY` | summariser back to plain greedy | off |

**Never run two at once.** llama-server is `--parallel 1`, so they serialise
and starve each other; one run went from ~90 s to 24 minutes and the timings
from both were worthless.

## The projects

`scenario_project.py` builds **csvstats** — ~300 lines with one seeded
off-by-one, its own git history created *after* the bug is seeded, so
`git checkout` restores the broken state and reveals nothing. Tasks: `fix`,
`audit`, `refactor`.

`ml_project.py` builds **churnkit** — a spec and an empty package. The agent
implements preprocessing, metrics from first principles, gradient-descent
logistic regression, k-fold CV and a CLI. It is graded by 13 tests it never
sees, copied in only after it stops: metrics checked against scikit-learn to
1e-6, ≥0.85 held-out accuracy, and `sklearn` absent from the package.

## The grader must be validated before it is believed

    python3 scenarios/validate_grader.py     # expects 13/13

`reference/churnkit` is an implementation written to satisfy the spec exactly.
A fault in a hidden test is invisible — the run still produces a number, and
the number means nothing. Re-run this after any change to `SPEC` or
`ACCEPTANCE`.

## One run is not a number

Four runs of the ML task on identical inputs scored **12, 10, 9, 9** out of 13.
The spread is entirely model-side; the harness metrics (refusals, stops,
compaction cost) stayed flat across all four. Quote a distribution, not a best
score.
