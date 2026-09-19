"""The synthetic task's metric. Lower is better - deliberately.

RMSE is the cheapest way to prove the policy layer is not quietly assuming
that bigger numbers win. Everything in agent/state.py compares in the
maximize direction, so a minimize task exercises the orientation boundary on
every single comparison rather than in a special case.

`quantum` rounds the metric onto a grid, simulating a discrete metric such as
accuracy over a small evaluation set, where a "gain" finer than one row is not
a gain at all.
"""
import math


def rmse(y_true, y_pred):
    n = len(y_true)
    if n == 0 or n != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n)


def evaluate(y_true, y_pred, quantum=0.0):
    """Returns {'RMSE':…, 'MAE':…, 'primary':…}. primary is RMSE, minimized."""
    n = len(y_true)
    r = rmse(y_true, y_pred)
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    if quantum:
        r = round(r / quantum) * quantum
    return {"RMSE": r, "MAE": mae, "primary": r, "rows": n, "groups": n}
