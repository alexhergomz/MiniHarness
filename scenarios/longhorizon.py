"""Long-horizon runs against a disposable project, with compaction timed.

The agent works on `csvstats` — a synthetic codebase generated fresh for each
run, in its own directory, with its own git history. It is never the harness's
own source and never anything of the user's. Earlier versions ran against a
copy of MiniHarness sitting beside the real repository; when the agent wrecked
a file it tried to `git checkout` the original to recover.

  MH_CTX=131072 MH_TASK=audit MH_DRY=0.8 python3 longhorizon.py

  MH_CTX          context window the harness budgets for  (default 131072)
  MH_TURNS        tool-round cap                          (default 45)
  MH_WORK         where to build it; must be disposable   (default /tmp/…)
  MH_TASK         fix | audit | refactor | ml              (default fix)
  MH_DRY          DRY sampler multiplier, unset = off
  MH_DRY_LAST     DRY lookback in tokens                  (default 512)
  MH_NOPENALTY    1 = summariser back to plain greedy
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)          # this file now lives in the repository
# Disposable by construction: rebuilt from this directory before every run,
# and `workdir.disposable_or_die` refuses anywhere it would not be safe to
# delete — the repository, your home, or anywhere outside a temp root.
WORK = os.environ.get("MH_WORK", "/tmp/mh-scenario-csvstats")

sys.path.insert(0, HERE)
import ml_project
import scenario_project
import workdir


def suite(where):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=where,
                       capture_output=True, text=True, timeout=900)
    tail = [l for l in r.stdout.strip().splitlines()
            if " passed" in l or " failed" in l or " error" in l]
    return r.returncode, tail[-1] if tail else r.stdout.strip()[-200:]


def main():
    task_name = os.environ.get("MH_TASK", "fix")
    ml = task_name == "ml"
    if ml:
        # A build-from-spec task. There is no seeded bug and no starting suite;
        # the grade comes from an acceptance suite the agent never sees, copied
        # in only after it has stopped.
        work = ml_project.build(WORK)
        print(f"ML project scaffolded at {work} (spec only, no implementation)")
    else:
        work = scenario_project.build(WORK)
        rc, line = suite(work)
        print(f"scenario built at {work}: {line}")
        if rc == 0:
            sys.exit("the seeded bug did not break the suite")

    sys.path.insert(0, SRC)
    from miniharness import config as cfg_mod, context, loop, provider

    config = cfg_mod.load()
    config.update({
        "_cwd": work,
        "model": "local",
        "llama_host": "127.0.0.1",
        "llama_port": int(os.environ.get("MH_PORT", "8890")),
        "llama_ctx": int(os.environ.get("MH_CTX", "131072")),
        "accept_all": True,
        "max_turns": int(os.environ.get("MH_TURNS", "45")),
        "autostart": False,
    })

    # MH_QWEN=1 sends Qwen's own documented sampling for Qwen3.5 thinking mode
    # (coding profile): temperature 0.6, top_k 20, min_p 0.0. The harness
    # default is 0.3/40/0.05 — every value off-spec, and temperature at half
    # the recommended minimum, which is the classic driver of degenerate
    # repetition. MH_PRESENCE sets presence_penalty, the lever Qwen ships at
    # 1.5 for one of their reasoning variants.
    if os.environ.get("MH_QWEN") or os.environ.get("MH_PRESENCE"):
        real_connect_q = provider._connect
        pres = float(os.environ.get("MH_PRESENCE", "0"))
        spec = bool(os.environ.get("MH_QWEN"))

        def connect_qwen(url, headers, payload, cfg):
            if spec:
                payload.update(temperature=0.6, top_k=20, min_p=0.0, top_p=0.95)
            if pres:
                payload["presence_penalty"] = pres
            return real_connect_q(url, headers, payload, cfg)

        provider._connect = connect_qwen
        print(f"Qwen sampling: spec={spec} presence_penalty={pres}")

    if os.environ.get("MH_DRY"):
        # Injected rather than configured: the harness should not carry a
        # sampler setting until it is measured. DRY penalises reproducing a
        # sequence already emitted, which is what a degenerate loop is — but
        # the lookback must stay bounded. At -1 (whole context) it also
        # penalises quoting the prompt, and Edit exists to quote the prompt:
        # measured, three edits drifted off the exact text and the model then
        # overwrote a 1,002-line module with a fragment.
        real_connect = provider._connect
        dry = float(os.environ["MH_DRY"])
        last = int(os.environ.get("MH_DRY_LAST", "512"))

        def connect_with_dry(url, headers, payload, cfg):
            payload.update(dry_multiplier=dry, dry_base=1.75,
                           dry_allowed_length=4, dry_penalty_last_n=last)
            return real_connect(url, headers, payload, cfg)

        provider._connect = connect_with_dry
        print(f"DRY on: multiplier={dry}, lookback={last} tokens")

    if os.environ.get("MH_NOPENALTY"):
        real_stream = provider.stream

        def unpenalised(model, system, messages, schemas, cfg, *a, **k):
            if cfg.get("frequency_penalty"):
                cfg = dict(cfg)
                cfg.pop("frequency_penalty")
            return real_stream(model, system, messages, schemas, cfg, *a, **k)

        provider.stream = unpenalised
        print("A/B: summariser penalty DISABLED")

    stats = {"compactions": [], "summaries": []}
    real_compact, real_summarise = loop._compact_if_needed, context.summarise_span

    def timed_compact(state, cfg, schemas=None):
        t0, before = time.monotonic(), len(stats["summaries"])
        freed = real_compact(state, cfg, schemas)
        if freed:
            spent = time.monotonic() - t0
            gen = sum(stats["summaries"][before:])
            stats["compactions"].append((freed, spent, gen))
            print(f"      [compaction] reclaimed ~{freed} tok in {spent:.1f}s "
                  f"({gen:.1f}s of it summarising)")
        return freed

    def timed_summarise(*a, **k):
        t0 = time.monotonic()
        out = real_summarise(*a, **k)
        stats["summaries"].append(time.monotonic() - t0)
        if out:
            print(f"      [note] {len(out)} chars in {stats['summaries'][-1]:.1f}s")
        return out

    loop._compact_if_needed = timed_compact
    context.summarise_span = timed_summarise

    tracker = context.FileTracker()
    state = loop.State(system=context.build_system(config, work))
    state.add_user({"audit": scenario_project.AUDIT_TASK,
                    "refactor": scenario_project.REFACTOR_TASK,
                    "ml": ml_project.TASK}.get(task_name, scenario_project.TASK))

    # A turn that calls no tool is otherwise invisible, and a runaway lives only
    # in memory until the turn ends — killing the run to look destroys the only
    # copy.
    # Per-run, not a fixed name: two runs sharing one dump silently interleave
    # their streams, and the second one's evidence looks like the first one's.
    # Into the work directory, not next to this file: the dump describes one
    # run of a disposable project and dies with it. Nothing a run produces is
    # written into the repository.
    dump = open(os.path.join(work, os.environ.get(
        "MH_DUMP", f"gen_stream_{task_name}.log")), "w", buffering=1)

    print(f"\nrunning: ctx={config['llama_ctx']} max_turns={config['max_turns']}\n")
    t0 = time.monotonic()
    calls = refused = 0
    edits = {"total": 0, "failed": 0, "recovered": 0, "loose_applied": 0}
    written = {"think": 0, "text": 0}
    beat = [time.monotonic()]
    turn_start = [time.monotonic()]

    for ev in loop.run(state, config, tracker=tracker):
        kind = type(ev).__name__
        if kind in ("ThinkChunk", "TextChunk"):
            text = getattr(ev, "text", "")
            dump.write(text)
            written["think" if kind == "ThinkChunk" else "text"] += len(text)
            now = time.monotonic()
            if now - beat[0] >= 30:
                beat[0] = now
                print(f"      [thinking] ~{written['think'] // 4:,} tokens, "
                      f"{now - turn_start[0]:.0f}s")
        elif kind == "ToolStart":
            calls += 1
            dump.write(f"\n\n===== after this: CALL #{calls} {ev.name} "
                       f"[think {written['think']}B / text {written['text']}B] =====\n\n")
            written = {"think": 0, "text": 0}
            turn_start[0] = beat[0] = time.monotonic()
            print(f"   #{calls:>3} [t+{time.monotonic() - t0:>6.0f}s] "
                  f"{ev.name} {str(ev.params)[:100]}")
        elif kind == "ToolEnd":
            res = str(getattr(ev, "result", ""))
            if res.startswith("Error: refused"):
                refused += 1
                print(f"      [refused] {ev.result[:120]}")
            if ev.name == "Edit":
                edits["total"] += 1
                if res.startswith("Error"):
                    edits["failed"] += 1
                    edits["pending"] = True
                    kind_of = ("not-read-yet" if "had not been read" in res
                               else "not-found-with-hint" if "Did you mean" in res
                               else "not-found" if "not found" in res
                               else "ambiguous" if "appears" in res
                               else "refused" if "refused" in res
                               else "other")
                    edits[kind_of] = edits.get(kind_of, 0) + 1
                    print(f"      [edit failed: {kind_of}]")
                else:
                    if edits.pop("pending", False):
                        edits["recovered"] += 1
                        print("      [edit recovered after a failure]")
                    if "ignoring whitespace" in res:
                        edits["loose_applied"] += 1
        elif kind == "Notice":
            print(f"      [notice] {ev.text}")
    elapsed = time.monotonic() - t0

    comp = stats["compactions"]
    print(f"\n{'=' * 70}")
    if ml:
        passed, total, tail = ml_project.grade(work)
        print(f"verdict     : {passed}/{total} acceptance tests "
              f"({'COMPLETE' if total and passed == total else 'partial'})")
        if tail:
            print("failing     :")
            for line in tail.splitlines()[:12]:
                print(f"   {line[:110]}")
        rc, own = suite(work)
        print(f"own tests   : {own}")
    else:
        rc, line = suite(work)
        print(f"verdict     : {'PASS' if rc == 0 else 'FAIL'}  ({line})")
    print(f"wall clock  : {elapsed:.0f}s over {calls} tool calls")
    print(f"refused     : {refused} (jail or deny-list)")
    if edits["total"]:
        print(f"edits       : {edits['total']} attempted, {edits['failed']} failed, "
              f"{edits['recovered']} recovered after a failure, "
              f"{edits['loose_applied']} applied via whitespace match")
        for k in ("not-read-yet", "not-found-with-hint", "not-found",
                  "ambiguous", "refused", "other"):
            if edits.get(k):
                print(f"   {k:<22} {edits[k]}")
    if comp:
        total = sum(c[1] for c in comp)
        print(f"compactions : {len(comp)}, {total:.0f}s total "
              f"({100 * total / elapsed:.0f}% of wall clock), "
              f"{sum(c[2] for c in comp):.0f}s summarising")
    else:
        print("compactions : none")


if __name__ == "__main__":
    main()
