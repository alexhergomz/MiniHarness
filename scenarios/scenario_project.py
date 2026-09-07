"""A disposable codebase for long-horizon runs. Nothing of the user's in it.

The scenarios used to run against a copy of MiniHarness itself, sitting beside
the real repository. That was wrong twice over: the agent under test was being
asked to work on the harness's own source, and when it wrecked a file it tried
to `git checkout` the original repo to recover. Neither the code nor the
location should ever have been the user's.

So: a synthetic project, generated fresh, in a directory of its own, with its
own git history created *after* the bug is seeded — so `git checkout` restores
the broken state and reveals nothing about the intended fix.
"""
import os, shutil, subprocess, textwrap

import workdir

FILES: dict[str, str] = {}

FILES["csvstats/__init__.py"] = '''"""Summary statistics for delimited files."""

__version__ = "0.3.1"
'''

FILES["csvstats/parser.py"] = '''"""Reading delimited text into rows of typed values."""


class ParseError(ValueError):
    """Raised when a row cannot be read under the current dialect."""


def sniff_delimiter(sample: str) -> str:
    """Guess the delimiter from the first non-empty line."""
    for line in sample.splitlines():
        if not line.strip():
            continue
        counts = {d: line.count(d) for d in (",", ";", "\\t", "|")}
        best = max(counts, key=counts.get)
        return best if counts[best] else ","
    return ","


def coerce(value: str):
    """Text to int, float, or text — in that order."""
    v = value.strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse(text: str, delimiter: str | None = None) -> tuple[list[str], list[list]]:
    """Return (header, rows). The first line is always the header."""
    delimiter = delimiter or sniff_delimiter(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ParseError("no data")
    header = [h.strip() for h in lines[0].split(delimiter)]
    rows = []
    for n, line in enumerate(lines[1:], start=2):
        cells = line.split(delimiter)
        if len(cells) != len(header):
            raise ParseError(
                f"line {n}: expected {len(header)} fields, got {len(cells)}")
        rows.append([coerce(c) for c in cells])
    return header, rows


def column(header: list[str], rows: list[list], name: str) -> list:
    """Every value in one column, missing cells dropped."""
    if name not in header:
        raise KeyError(f"no column named {name!r}; have {', '.join(header)}")
    i = header.index(name)
    return [r[i] for r in rows if r[i] is not None]
'''

FILES["csvstats/stats.py"] = '''"""The statistics themselves. Pure functions over lists of numbers."""


def _numeric(values):
    out = [v for v in values if isinstance(v, (int, float))]
    if not out:
        raise ValueError("no numeric values")
    return sorted(out)


def mean(values):
    v = _numeric(values)
    return sum(v) / len(v)


def median(values):
    v = _numeric(values)
    mid = len(v) // 2
    if len(v) % 2:
        return v[mid]
    return (v[mid - 1] + v[mid]) / 2


def percentile(values, p):
    """The p-th percentile, 0 <= p <= 100, by nearest rank."""
    if not 0 <= p <= 100:
        raise ValueError("percentile must be between 0 and 100")
    v = _numeric(values)
    rank = (p / 100) * (len(v) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(v) - 1)
    frac = rank - lo
    return v[lo] * (1 - frac) + v[hi] * frac


def spread(values):
    """(min, max) of the numeric values."""
    v = _numeric(values)
    return v[0], v[-1]


def stdev(values):
    v = _numeric(values)
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def summary(values) -> dict:
    """Everything, in one pass, for the report layer."""
    return {
        "count": len(_numeric(values)),
        "mean": mean(values),
        "median": median(values),
        "p90": percentile(values, 90),
        "min": spread(values)[0],
        "max": spread(values)[1],
        "stdev": stdev(values),
    }
'''

FILES["csvstats/report.py"] = '''"""Turning a summary into something readable."""

ORDER = ["count", "mean", "median", "p90", "min", "max", "stdev"]


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def as_table(name: str, summary: dict) -> str:
    width = max(len(k) for k in ORDER)
    lines = [f"{name}", "-" * len(name)]
    for key in ORDER:
        if key in summary:
            lines.append(f"  {key:<{width}}  {fmt(summary[key])}")
    return "\\n".join(lines)


def as_csv(name: str, summary: dict) -> str:
    keys = [k for k in ORDER if k in summary]
    return ",".join(["column"] + keys) + "\\n" + \\
           ",".join([name] + [fmt(summary[k]) for k in keys])
'''

FILES["csvstats/cli.py"] = '''"""Command line entry point."""

import argparse
import sys

from . import parser as parsing
from . import report, stats


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="csvstats",
                                 description="Summary statistics for CSV files")
    ap.add_argument("path", help="file to read, or - for stdin")
    ap.add_argument("-c", "--column", action="append", default=[],
                    help="column to summarise (repeatable; default: all numeric)")
    ap.add_argument("-d", "--delimiter", default=None)
    ap.add_argument("--csv", action="store_true", help="emit CSV, not a table")
    return ap


def numeric_columns(header, rows):
    out = []
    for name in header:
        values = parsing.column(header, rows, name)
        if any(isinstance(v, (int, float)) for v in values):
            out.append(name)
    return out


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    text = sys.stdin.read() if args.path == "-" else open(args.path).read()
    try:
        header, rows = parsing.parse(text, args.delimiter)
    except parsing.ParseError as e:
        print(f"csvstats: {e}", file=sys.stderr)
        return 2
    wanted = args.column or numeric_columns(header, rows)
    chunks = []
    for name in wanted:
        values = parsing.column(header, rows, name)
        summary = stats.summary(values)
        chunks.append((report.as_csv if args.csv else report.as_table)(name, summary))
    print("\\n\\n".join(chunks))
    return 0
'''

FILES["tests/test_parser.py"] = '''from csvstats import parser


def test_sniffs_a_comma():
    assert parser.sniff_delimiter("a,b,c\\n1,2,3\\n") == ","


def test_sniffs_a_semicolon():
    assert parser.sniff_delimiter("a;b;c\\n1;2;3\\n") == ";"


def test_coerces_types():
    assert parser.coerce(" 4 ") == 4
    assert parser.coerce("4.5") == 4.5
    assert parser.coerce("hello") == "hello"
    assert parser.coerce("  ") is None


def test_parses_a_table():
    header, rows = parser.parse("name,size\\nwidget,3\\ngasket,7\\n")
    assert header == ["name", "size"]
    assert rows == [["widget", 3], ["gasket", 7]]


def test_a_short_row_is_an_error():
    import pytest
    with pytest.raises(parser.ParseError):
        parser.parse("a,b\\n1\\n")


def test_column_drops_missing():
    header, rows = parser.parse("a,b\\n1,2\\n,4\\n")
    assert parser.column(header, rows, "a") == [1]
'''

FILES["tests/test_stats.py"] = '''import pytest

from csvstats import stats


def test_mean_and_median():
    assert stats.mean([1, 2, 3]) == 2
    assert stats.median([1, 3, 2]) == 2
    assert stats.median([1, 2, 3, 4]) == 2.5


def test_spread_and_stdev():
    assert stats.spread([5, 1, 9]) == (1, 9)
    assert stats.stdev([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.13809, rel=1e-4)


def test_percentile_endpoints():
    v = list(range(1, 11))          # 1..10
    assert stats.percentile(v, 0) == 1
    assert stats.percentile(v, 100) == 10


def test_percentile_interpolates():
    v = list(range(1, 11))
    assert stats.percentile(v, 50) == pytest.approx(5.5)
    assert stats.percentile(v, 90) == pytest.approx(9.1)


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        stats.percentile([1, 2], 101)


def test_summary_has_every_field():
    s = stats.summary([1, 2, 3, 4, 5])
    assert set(s) == {"count", "mean", "median", "p90", "min", "max", "stdev"}
'''

FILES["tests/test_report.py"] = '''from csvstats import report


def test_table_lists_every_key():
    text = report.as_table("size", {"count": 3, "mean": 2.0, "median": 2,
                                    "p90": 2.8, "min": 1, "max": 3,
                                    "stdev": 1.0})
    for key in report.ORDER:
        assert key in text


def test_floats_are_trimmed():
    assert report.fmt(2.500) == "2.5"
    assert report.fmt(3.0) == "3"
    assert report.fmt(7) == "7"


def test_csv_has_a_header_and_a_row():
    out = report.as_csv("size", {"count": 2, "mean": 1.5})
    lines = out.splitlines()
    assert lines[0].startswith("column,")
    assert lines[1].startswith("size,")
'''

FILES["tests/test_cli.py"] = '''import io
import sys

from csvstats import cli


def test_numeric_columns_ignores_text():
    from csvstats import parser
    header, rows = parser.parse("name,size\\nwidget,3\\n")
    assert cli.numeric_columns(header, rows) == ["size"]


def test_main_reads_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("a,b\\n1,2\\n3,4\\n"))
    assert cli.main(["-"]) == 0
    assert "mean" in capsys.readouterr().out


def test_a_ragged_file_exits_two(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("a,b\\n1\\n"))
    assert cli.main(["-"]) == 2
'''

FILES["pyproject.toml"] = '''[project]
name = "csvstats"
version = "0.3.1"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
'''

FILES["README.md"] = '''# csvstats

Summary statistics for delimited files.

    csvstats data.csv
    csvstats -c size --csv data.csv
    cat data.csv | csvstats -

Layout: `parser.py` reads text into rows, `stats.py` computes, `report.py`
formats, `cli.py` wires them together.
'''

# The seeded defect. `hi` clamps to the last index, so at p=100 `lo` is already
# the last element and the interpolation reads one past the end of the window
# it should have used — off by one in the same family as the real bug this
# scenario is modelled on, and caught by an existing test rather than a new one.
SEED = ("csvstats/stats.py",
        "    hi = min(lo + 1, len(v) - 1)",
        "    hi = min(lo + 1, len(v))")


def build(where: str) -> str:
    """Materialise the project at `where`, seeded and committed. Returns path."""
    where = workdir.disposable_or_die(where)
    if os.path.exists(where):
        shutil.rmtree(where)
    for rel, body in FILES.items():
        path = os.path.join(where, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(body)

    path = os.path.join(where, SEED[0])
    with open(path) as f:
        text = f.read()
    if SEED[1] not in text:
        raise SystemExit(f"seed anchor missing from {SEED[0]}")
    with open(path, "w") as f:
        f.write(text.replace(SEED[1], SEED[2]))

    # Git history created *after* seeding: `git checkout` restores the broken
    # state, so the agent can undo its own mistakes without the history ever
    # revealing what the fix is meant to be.
    env = dict(os.environ, GIT_AUTHOR_NAME="scenario", GIT_AUTHOR_EMAIL="s@local",
               GIT_COMMITTER_NAME="scenario", GIT_COMMITTER_EMAIL="s@local")
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "commit", "-qm", "csvstats 0.3.1"]):
        subprocess.run(cmd, cwd=where, env=env, check=True,
                       capture_output=True)
    return where


TASK = (
    "`python3 -m pytest -q` has one failing test in this repository. Find what "
    "is actually broken and fix it. Then check whether the same class of "
    "mistake appears anywhere else in csvstats/, and add a test for anything "
    "you find. The whole suite must pass when you are done."
)

# Many edits to existing text, which is what exercises verbatim quoting and so
# the "did you mean" recovery path. The audit task produces exactly one edit —
# the fix — and cannot measure it.
REFACTOR_TASK = (
    "`python3 -m pytest -q` has one failing test in this repository. Fix it "
    "first. Then rename `percentile` to `quantile` throughout: the definition "
    "in csvstats/stats.py, every call site in csvstats/, and every test that "
    "refers to it. Keep the behaviour identical. The whole suite must pass "
    "when you are done, and `grep -rn percentile csvstats tests` must return "
    "nothing."
)

AUDIT_TASK = (
    "`python3 -m pytest -q` has one failing test in this repository. Fix it "
    "first. Then audit every module in csvstats/ for the same class of mistake "
    "— an index or slice computed from the wrong variable — and write what you "
    "find to AUDIT.md, one line per module in the form `<module>: <finding or "
    "'clean'>`. Every module must appear. The whole suite must still pass."
)
