import numpy as np

def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos]); out[~pos] = e / (1.0 + e)
    return out

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=500, l2=0.0, seed=0):
        self.lr, self.epochs, self.l2, self.seed = lr, epochs, l2, seed
        self.w_ = None; self.b_ = 0.0; self.loss_history_ = []
    def fit(self, X, y):
        A = np.asarray(X, dtype=float); t = np.asarray(y, dtype=float)
        n, d = A.shape
        self.w_ = np.zeros(d); self.b_ = 0.0; self.loss_history_ = []
        for _ in range(self.epochs):
            p = _sigmoid(A @ self.w_ + self.b_)
            eps = 1e-12
            loss = -np.mean(t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps))
            loss += self.l2 * np.sum(self.w_ ** 2) / 2
            self.loss_history_.append(float(loss))
            err = p - t
            self.w_ -= self.lr * (A.T @ err / n + self.l2 * self.w_)
            self.b_ -= self.lr * err.mean()
        return self
    def predict_proba(self, X):
        p = _sigmoid(np.asarray(X, dtype=float) @ self.w_ + self.b_)
        return np.clip(p, 1e-15, 1 - 1e-15)
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
