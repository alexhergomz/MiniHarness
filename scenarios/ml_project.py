"""A formidable ML build task, and a hidden suite that grades it.

The agent is given SPEC.md and an empty package. It must implement a tabular
classification toolkit from scratch — preprocessing, a logistic regression
trained by gradient descent, metrics, k-fold cross-validation and a CLI.

The acceptance tests live here, outside the jail, and are copied in only after
the agent has stopped. It never sees them: it is graded on whether the code
does what the spec says, not on whether it satisfied tests it could read. The
spec is precise about names and signatures so that grading is objective rather
than a matter of interpretation.

Deliberately harder than anything this harness has been run on: ~6 modules, a
numerical algorithm that has to actually converge, and metrics that are checked
against scikit-learn to 1e-6.
"""
import os, shutil, subprocess

import workdir

SPEC = '''# churnkit — a tabular classification toolkit

Build a small, dependency-light toolkit for binary classification on tabular
data. `numpy` and `pandas` are available and expected. **`scikit-learn` must
not be imported by `churnkit/` at all** — the estimator and the metrics are the
exercise. (You may use it in your own scratch checks if you wish.)

Every name and signature below is part of the contract and will be checked.

## churnkit/data.py

    make_dataset(n: int = 1000, n_features: int = 6, noise: float = 0.1,
                 seed: int = 0) -> tuple[pandas.DataFrame, numpy.ndarray]

Deterministic for a given seed. Returns `(X, y)` where `X` is a DataFrame with
`n` rows and `n_features` columns named `f0 … f{n-1}`, and `y` is an integer
array of 0/1 with both classes present. The relationship must be learnable: a
linear decision boundary plus `noise`.

    train_test_split(X, y, test_size: float = 0.25, seed: int = 0)
        -> tuple[X_train, X_test, y_train, y_test]

Deterministic. No row appears in both parts, and every row appears in one.

## churnkit/features.py

    class StandardScaler:
        fit(self, X) -> "StandardScaler"
        transform(self, X) -> numpy.ndarray
        fit_transform(self, X) -> numpy.ndarray

Column-wise standardisation to mean 0, standard deviation 1, using the
statistics learned in `fit`. A column with zero variance must not produce NaN.
`transform` before `fit` raises `RuntimeError`.

## churnkit/metrics.py

Implement from first principles, each taking `(y_true, y_pred)` as 1-D arrays:

    accuracy(y_true, y_pred) -> float
    precision(y_true, y_pred) -> float
    recall(y_true, y_pred) -> float
    f1(y_true, y_pred) -> float
    confusion_matrix(y_true, y_pred) -> numpy.ndarray   # 2x2, [[tn, fp], [fn, tp]]
    roc_auc(y_true, y_score) -> float                   # y_score is a probability

`precision`, `recall` and `f1` return 0.0 rather than dividing by zero.

## churnkit/model.py

    class LogisticRegression:
        __init__(self, lr: float = 0.1, epochs: int = 500,
                 l2: float = 0.0, seed: int = 0)
        fit(self, X, y) -> "LogisticRegression"
        predict_proba(self, X) -> numpy.ndarray     # P(y=1), values in (0, 1)
        predict(self, X, threshold: float = 0.5) -> numpy.ndarray
        loss_history_: list[float]                  # mean loss per epoch, after fit

Full-batch gradient descent on the mean binary cross-entropy, with optional L2
on the weights but never on the intercept. `loss_history_` must be
non-increasing over the last 80% of epochs — if it is not, the learning rate or
the gradient is wrong. Guard the sigmoid against overflow.

## churnkit/validate.py

    cross_val_score(model_factory, X, y, k: int = 5, seed: int = 0) -> list[float]

`model_factory` is a zero-argument callable returning a fresh model. Returns
`k` accuracy scores. Folds must be disjoint and cover every row. **Scaling must
be fitted on the training folds only** — fitting it on all the data before
splitting leaks the test statistics and is wrong.

## churnkit/cli.py

    main(argv: list[str] | None = None) -> int

    churnkit train --rows 2000 --folds 5 --epochs 300

Generates data, runs cross-validation, prints one line per fold as
`fold <i>: accuracy=<value>` and a final `mean accuracy=<value>`. Returns 0 on
success, 2 on bad arguments.

## What "done" means

- `python3 -m pytest -q` passes, and you have written tests of your own.
- On `make_dataset(2000, seed=0)` with a 75/25 split, the model reaches at
  least **0.85** accuracy on the held-out part.
- No module under `churnkit/` imports `sklearn`.
'''

ACCEPTANCE = r'''"""Hidden acceptance suite. The agent never sees this file."""
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("churnkit")


def test_no_sklearn_in_the_package():
    import pathlib
    for p in pathlib.Path("churnkit").rglob("*.py"):
        assert "sklearn" not in p.read_text(), f"{p} imports sklearn"


def test_dataset_is_deterministic_and_balanced():
    from churnkit.data import make_dataset
    X1, y1 = make_dataset(500, seed=3)
    X2, y2 = make_dataset(500, seed=3)
    assert list(X1.columns) == [f"f{i}" for i in range(X1.shape[1])]
    assert X1.shape == (500, 6)
    assert np.array_equal(np.asarray(y1), np.asarray(y2))
    assert set(np.unique(np.asarray(y1))) == {0, 1}


def test_split_is_a_partition():
    from churnkit.data import make_dataset, train_test_split
    X, y = make_dataset(200, seed=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=1)
    assert len(Xtr) + len(Xte) == 200
    assert abs(len(Xte) - 50) <= 1
    assert len(ytr) == len(Xtr) and len(yte) == len(Xte)


def test_scaler_standardises_and_reuses_its_statistics():
    from churnkit.features import StandardScaler
    rng = np.random.default_rng(0)
    A = rng.normal(5, 3, size=(300, 4))
    s = StandardScaler().fit(A)
    Z = s.transform(A)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-6)
    # A second matrix must be scaled by the FIRST one's statistics.
    B = A + 10
    assert np.allclose(s.transform(B), Z + 10 / A.std(axis=0), atol=1e-6)


def test_scaler_survives_a_constant_column():
    from churnkit.features import StandardScaler
    A = np.column_stack([np.ones(50), np.arange(50.0)])
    Z = StandardScaler().fit_transform(A)
    assert np.isfinite(Z).all()


def test_transform_before_fit_raises():
    from churnkit.features import StandardScaler
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.zeros((3, 2)))


def test_metrics_match_sklearn():
    from sklearn import metrics as sk
    from churnkit import metrics as m
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 400)
    p = rng.integers(0, 2, 400)
    s = rng.random(400)
    assert m.accuracy(y, p) == pytest.approx(sk.accuracy_score(y, p), abs=1e-9)
    assert m.precision(y, p) == pytest.approx(sk.precision_score(y, p), abs=1e-9)
    assert m.recall(y, p) == pytest.approx(sk.recall_score(y, p), abs=1e-9)
    assert m.f1(y, p) == pytest.approx(sk.f1_score(y, p), abs=1e-9)
    assert np.array_equal(np.asarray(m.confusion_matrix(y, p)),
                          sk.confusion_matrix(y, p))
    assert m.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-6)


def test_metrics_do_not_divide_by_zero():
    from churnkit import metrics as m
    y = np.zeros(10, dtype=int)
    p = np.zeros(10, dtype=int)
    assert m.precision(y, p) == 0.0
    assert m.recall(y, p) == 0.0
    assert m.f1(y, p) == 0.0


def test_model_converges_and_records_its_loss():
    from churnkit.data import make_dataset
    from churnkit.features import StandardScaler
    from churnkit.model import LogisticRegression
    X, y = make_dataset(800, seed=2)
    Z = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    mdl = LogisticRegression(lr=0.1, epochs=300).fit(Z, y)
    hist = list(mdl.loss_history_)
    assert len(hist) == 300
    tail = hist[len(hist) // 5:]
    assert all(b <= a + 1e-6 for a, b in zip(tail, tail[1:])), "loss went up"
    proba = np.asarray(mdl.predict_proba(Z))
    assert proba.shape == (800,)
    assert ((proba > 0) & (proba < 1)).all()


def test_model_generalises():
    from churnkit.data import make_dataset, train_test_split
    from churnkit.features import StandardScaler
    from churnkit.metrics import accuracy
    from churnkit.model import LogisticRegression
    X, y = make_dataset(2000, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0)
    s = StandardScaler().fit(np.asarray(Xtr, dtype=float))
    mdl = LogisticRegression(lr=0.1, epochs=500).fit(
        s.transform(np.asarray(Xtr, dtype=float)), ytr)
    acc = accuracy(np.asarray(yte),
                   np.asarray(mdl.predict(s.transform(np.asarray(Xte, dtype=float)))))
    assert acc >= 0.85, f"held-out accuracy {acc:.3f}"


def test_extreme_inputs_do_not_overflow():
    from churnkit.model import LogisticRegression
    X = np.array([[-800.0], [800.0]] * 20)
    y = np.array([0, 1] * 20)
    mdl = LogisticRegression(lr=0.5, epochs=50).fit(X, y)
    p = np.asarray(mdl.predict_proba(X))
    assert np.isfinite(p).all(), "sigmoid overflowed"


def test_cross_val_returns_k_scores():
    from churnkit.data import make_dataset
    from churnkit.model import LogisticRegression
    from churnkit.validate import cross_val_score
    X, y = make_dataset(600, seed=4)
    scores = cross_val_score(lambda: LogisticRegression(epochs=200), X, y, k=5, seed=4)
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert np.mean(scores) >= 0.75, f"mean {np.mean(scores):.3f}"


def test_cli_runs_end_to_end():
    r = subprocess.run([sys.executable, "-m", "churnkit.cli", "train",
                        "--rows", "400", "--folds", "3", "--epochs", "50"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    assert "mean accuracy=" in r.stdout
    assert r.stdout.count("fold ") >= 3
'''

TASK = (
    "Read SPEC.md and implement it. It describes `churnkit`, a tabular "
    "classification toolkit: data generation and splitting, a StandardScaler, "
    "metrics implemented from first principles, a logistic regression trained "
    "by gradient descent, k-fold cross-validation, and a CLI.\n\n"
    "Every name and signature in the spec is a contract — implement them "
    "exactly as written. numpy and pandas are available; `churnkit/` must not "
    "import sklearn. Write tests of your own as you go and keep "
    "`python3 -m pytest -q` passing. Work module by module rather than trying "
    "to write everything at once."
)


def build(where: str) -> str:
    """Materialise the empty project and its spec. Returns the path."""
    where = workdir.disposable_or_die(where)
    if os.path.exists(where):
        shutil.rmtree(where)
    os.makedirs(os.path.join(where, "churnkit"))
    os.makedirs(os.path.join(where, "tests"))

    with open(os.path.join(where, "SPEC.md"), "w") as f:
        f.write(SPEC)
    with open(os.path.join(where, "churnkit", "__init__.py"), "w") as f:
        f.write('"""churnkit — tabular classification toolkit."""\n\n'
                '__version__ = "0.1.0"\n')
    with open(os.path.join(where, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "churnkit"\nversion = "0.1.0"\n'
                'requires-python = ">=3.10"\n\n'
                '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')

    env = dict(os.environ, GIT_AUTHOR_NAME="scenario", GIT_AUTHOR_EMAIL="s@local",
               GIT_COMMITTER_NAME="scenario", GIT_COMMITTER_EMAIL="s@local")
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "commit", "-qm", "spec only"]):
        subprocess.run(cmd, cwd=where, env=env, check=True, capture_output=True)
    return where


def grade(where: str) -> tuple[int, int, str]:
    """Copy the hidden suite in, run it, and report (passed, total, tail)."""
    path = os.path.join(where, "_acceptance_test.py")
    with open(path, "w") as f:
        f.write(ACCEPTANCE)
    try:
        r = subprocess.run([os.sys.executable, "-m", "pytest", "-q", "--no-header",
                            "-p", "no:cacheprovider", path],
                           cwd=where, capture_output=True, text=True, timeout=1800)
        out = r.stdout
        import re
        passed = int((re.search(r"(\d+) passed", out) or [0, 0])[1])
        failed = int((re.search(r"(\d+) failed", out) or [0, 0])[1])
        errors = int((re.search(r"(\d+) error", out) or [0, 0])[1])
        total = passed + failed + errors
        tail = [l for l in out.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")]
        return passed, total, "\n".join(tail[:12]) or out.strip()[-400:]
    finally:
        os.unlink(path)
