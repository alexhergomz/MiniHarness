import argparse, sys
import numpy as np
from .data import make_dataset
from .model import LogisticRegression
from .validate import cross_val_score

def main(argv=None):
    ap = argparse.ArgumentParser(prog="churnkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--rows", type=int, default=1000)
    tr.add_argument("--folds", type=int, default=5)
    tr.add_argument("--epochs", type=int, default=300)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    X, y = make_dataset(args.rows, seed=0)
    scores = cross_val_score(lambda: LogisticRegression(epochs=args.epochs), X, y, k=args.folds)
    for i, s in enumerate(scores):
        print(f"fold {i}: accuracy={s:.4f}")
    print(f"mean accuracy={np.mean(scores):.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
