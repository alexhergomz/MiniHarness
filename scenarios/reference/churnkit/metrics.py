import numpy as np

def _c(y_true, y_pred):
    yt = np.asarray(y_true).astype(int); yp = np.asarray(y_pred).astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum()); tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
    return tn, fp, fn, tp

def accuracy(y_true, y_pred):
    tn, fp, fn, tp = _c(y_true, y_pred)
    return (tp + tn) / max(1, tn + fp + fn + tp)

def precision(y_true, y_pred):
    _, fp, _, tp = _c(y_true, y_pred)
    return tp / (tp + fp) if tp + fp else 0.0

def recall(y_true, y_pred):
    _, _, fn, tp = _c(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else 0.0

def f1(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r else 0.0

def confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = _c(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]])

def roc_auc(y_true, y_score):
    yt = np.asarray(y_true).astype(int); s = np.asarray(y_score, dtype=float)
    pos, neg = (yt == 1).sum(), (yt == 0).sum()
    if pos == 0 or neg == 0: return 0.0
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float); sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]: j += 1
        ranks[order[i:j+1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[yt == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
