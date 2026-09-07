import numpy as np, pandas as pd

def make_dataset(n=1000, n_features=6, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    w = rng.normal(size=n_features)
    logit = X @ w + rng.normal(scale=noise, size=n)
    y = (logit > 0).astype(int)
    if len(np.unique(y)) < 2:
        y[0] = 1 - y[0]
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), y

def train_test_split(X, y, test_size=0.25, seed=0):
    n = len(X); rng = np.random.default_rng(seed)
    idx = rng.permutation(n); cut = int(round(n * (1 - test_size)))
    tr, te = idx[:cut], idx[cut:]
    Xa = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
    Xb = X.iloc[te] if hasattr(X, "iloc") else X[te]
    return Xa, Xb, np.asarray(y)[tr], np.asarray(y)[te]
