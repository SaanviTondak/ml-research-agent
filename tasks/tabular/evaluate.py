"""Metrics for the generic tabular task. This file defines the score.

Kept separate and checksummed for the same reason the KuaiRand task keeps the
organizer's evaluate.py untouched: a candidate that "improves" by reimplementing
the metric has not improved anything, and the agent is never told not to try.

Each metric returns a dict carrying the primary value plus whatever else is
cheap to report, and declares its direction so the policy layer can orient it.
"""
import math

# name -> (higher_is_better, needs_probabilities)
DIRECTION = {
    "rmse": (False, False),
    "mae": (False, False),
    "logloss": (False, True),
    "auc": (True, True),
    "accuracy": (True, True),
}


def _finite(v):
    return v == v and v not in (float("inf"), float("-inf"))


def rmse(y, p):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y, p)) / len(y))


def mae(y, p):
    return sum(abs(a - b) for a, b in zip(y, p)) / len(y)


def logloss(y, p, eps=1e-15):
    t = 0.0
    for a, b in zip(y, p):
        b = min(max(b, eps), 1 - eps)
        t += -(a * math.log(b) + (1 - a) * math.log(1 - b))
    return t / len(y)


def auc(y, p):
    """Mann-Whitney U with tie correction; equals roc_auc_score."""
    pairs = sorted(zip(p, y))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    npos = sum(l for _, l in pairs)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    srank = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (srank - npos * (npos + 1) / 2.0) / (npos * nneg)


def accuracy(y, p, threshold=0.5):
    return sum(1 for a, b in zip(y, p) if (b >= threshold) == (a >= 0.5)) / len(y)


FUNCS = {"rmse": rmse, "mae": mae, "logloss": logloss,
         "auc": auc, "accuracy": accuracy}


def evaluate(y_true, y_pred, metric="rmse"):
    if metric not in FUNCS:
        raise ValueError(f"unknown metric {metric!r}; have {sorted(FUNCS)}")
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if not y_true:
        raise ValueError("nothing to score")

    primary = FUNCS[metric](y_true, y_pred)
    out = {metric: primary, "primary": primary,
           "rows": len(y_true), "groups": len(y_true)}
    # A couple of cheap companions, because a secondary metric that moves while
    # the primary does not is exactly what a postmortem needs.
    if metric in ("rmse", "mae"):
        out["rmse"], out["mae"] = rmse(y_true, y_pred), mae(y_true, y_pred)
    elif metric in ("auc", "accuracy", "logloss"):
        out["auc"] = auc(y_true, y_pred)
    out["primary"] = primary
    return out
