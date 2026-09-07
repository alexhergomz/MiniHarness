import numpy as np
from .features import StandardScaler
from .metrics import accuracy

def cross_val_score(model_factory, X, y, k=5, seed=0):
    A = np.asarray(X, dtype=float); t = np.asarray(y)
    n = len(A); idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k); out = []
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        s = StandardScaler().fit(A[tr])
        m = model_factory().fit(s.transform(A[tr]), t[tr])
        out.append(float(accuracy(t[te], m.predict(s.transform(A[te])))))
    return out
