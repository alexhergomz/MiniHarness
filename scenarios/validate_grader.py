"""Check the hidden ML grader against a known-good implementation.

The `ml` scenario is graded by 13 tests the agent never sees, so a fault in
those tests is invisible: the run still produces a number, and the number is
meaningless. `reference/churnkit` is an implementation written to satisfy the
spec exactly. If it does not score 13/13, the grader is wrong, not the agent.

Run this after any change to SPEC or ACCEPTANCE in ml_project.py:

    python3 scenarios/validate_grader.py
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ml_project

WORK = os.environ.get("MH_WORK", "/tmp/mh-ml-reference-check")


def main() -> int:
    work = ml_project.build(WORK)          # refuses a non-disposable path
    src = os.path.join(HERE, "reference", "churnkit")
    for name in sorted(os.listdir(src)):
        if name.endswith(".py"):
            shutil.copy(os.path.join(src, name),
                        os.path.join(work, "churnkit", name))

    passed, total, tail = ml_project.grade(work)
    print(f"reference implementation scores {passed}/{total}")
    if passed == total and total == 13:
        print("grader validated")
        return 0
    print(tail)
    print("GRADER IS WRONG — a scenario score taken now would mean nothing")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
