import numpy as np

class StandardScaler:
    def __init__(self): self.mean_ = None; self.scale_ = None
    def fit(self, X):
        A = np.asarray(X, dtype=float)
        self.mean_ = A.mean(axis=0)
        s = A.std(axis=0); s[s == 0] = 1.0
        self.scale_ = s
        return self
    def transform(self, X):
        if self.mean_ is None: raise RuntimeError("fit before transform")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_
    def fit_transform(self, X): return self.fit(X).transform(X)
