"""Where a scenario is allowed to build, and where it is not.

Both generators start with `shutil.rmtree(where)`, so the working path is not
merely a place results land — it is a path this code deletes. A mistyped
`MH_WORK` is therefore not a bad run, it is data loss, and the loss lands on
whatever was already there.

The rule the scenarios are held to: **the agent works on destructible things
only.** What defines the experiment — the runner, the generators, the spec and
the hidden grader — lives in this repository under version control, which is
the backup. What the agent touches is regenerated from that source before every
run and can be deleted at any moment without costing anything.

The three refusals below are what makes that a rule rather than an intention.
They were written after the tooling itself spent a session in /tmp and was
destroyed four times: the durable half had been stored in the destructible
place, which is the same mistake in the other direction.
"""
import os
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Anywhere a system temp cleaner is entitled to delete without asking.
_TEMP_ROOTS = tuple({"/tmp", "/var/tmp", tempfile.gettempdir()})


def disposable_or_die(where: str) -> str:
    """Return `where` if it is safe to delete and rebuild, else exit."""
    path = os.path.abspath(os.path.expanduser(where))
    home = os.path.abspath(os.path.expanduser("~"))

    if path == os.path.sep or path.rstrip(os.sep) == "":
        raise SystemExit("refusing to build a scenario at the filesystem root")
    if _within(path, REPO):
        raise SystemExit(
            f"refusing to build a scenario inside the harness repository "
            f"({path}).\nThe agent must never be given the source that "
            f"defines the experiment — damage to it invalidates the result.")
    if _within(path, home):
        raise SystemExit(
            f"refusing to build a scenario under your home directory "
            f"({path}).\nSet MH_WORK to somewhere disposable, e.g. "
            f"/tmp/mh-scenario.")
    if not any(_within(path, root) for root in _TEMP_ROOTS):
        raise SystemExit(
            f"refusing to build a scenario outside a temp root ({path}).\n"
            f"Allowed: {', '.join(sorted(_TEMP_ROOTS))}. This directory is "
            f"deleted and regenerated on every run.")
    return path


def _within(path: str, root: str) -> bool:
    root = os.path.abspath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)
