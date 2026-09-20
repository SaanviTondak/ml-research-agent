"""A deliberately plain baseline: ridge / logistic regression on raw columns.

Two jobs. It is the thing the agent has to beat, and it is what
agent/calibrate.py runs on k seeds to measure the task's noise floor before
the search starts - so every task needs one, and it is worth having it be
boring, fast and correct rather than good.

It handles whatever the loader hands it: numeric columns straight through,
low-cardinality string columns one-hot encoded, missing values filled with the
training mean. That is enough to be a real baseline on most tabular problems
and leaves obvious headroom (interactions, better encodings, trees, tuning).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

HEADER = ["row_id", "score"]
MAX_LEVELS = 20          # above this a string column is dropped, not one-hot


def read_csv(path):
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        rows = [r for r in rd]
    return head, rows


def build_encoder(head, rows, target):
    """Decide, from the training split only, how each column becomes numbers."""
    idx = {c: i for i, c in enumerate(head)}
    plan = []
    for c in head:
        if c == target:
            continue
        vals = [r[idx[c]] for r in rows]
        numeric = True
        for v in vals:
            if v == "":
                continue
            try:
                float(v)
            except ValueError:
                numeric = False
                break
        if numeric:
            nums = [float(v) for v in vals if v != ""]
            plan.append(("num", c, idx[c], (sum(nums) / len(nums)) if nums else 0.0))
        else:
            levels = sorted(set(vals))
            if len(levels) <= MAX_LEVELS:
                plan.append(("cat", c, idx[c], levels))
    return plan


def encode(plan, rows):
    cols = []
    for kind, _c, i, extra in plan:
        if kind == "num":
            col = []
            for r in rows:
                v = r[i]
                col.append(float(v) if v != "" else extra)
            cols.append(np.asarray(col, dtype=float).reshape(-1, 1))
        else:
            lv = {v: k for k, v in enumerate(extra)}
            m = np.zeros((len(rows), len(extra)))
            for j, r in enumerate(rows):
                k = lv.get(r[i])
                if k is not None:
                    m[j, k] = 1.0
            cols.append(m)
    X = np.hstack(cols) if cols else np.zeros((len(rows), 0))
    return np.hstack([X, np.ones((len(X), 1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", required=True)
    ap.add_argument("--classification", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = Path(a.data_dir)
    head, tr = read_csv(d / "train.csv")
    ev_head, ev = read_csv(d / f"{a.split}.csv")
    ti = head.index(a.target)
    y = np.asarray([float(r[ti]) for r in tr])

    plan = build_encoder(head, tr, a.target)
    # The eval split may not carry the target column at all, so encode it
    # against its own header rather than assuming the two line up.
    ev_plan = [(k, c, ev_head.index(c), extra) for k, c, _i, extra in plan
               if c in ev_head]
    X, Xe = encode(plan, tr), encode(ev_plan, ev)
    if Xe.shape[1] != X.shape[1]:
        raise SystemExit(f"column mismatch: train {X.shape[1]}, "
                         f"{a.split} {Xe.shape[1]}")

    A = X.T @ X + a.alpha * np.eye(X.shape[1])
    if a.classification:
        # A few Newton steps of logistic regression; plenty for a baseline.
        w = np.zeros(X.shape[1])
        for _ in range(25):
            p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
            g = X.T @ (p - y) + a.alpha * w
            H = X.T @ (X * (p * (1 - p))[:, None]) + a.alpha * np.eye(len(w))
            try:
                w -= np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
        pred = 1.0 / (1.0 + np.exp(-np.clip(Xe @ w, -30, 30)))
    else:
        w = np.linalg.solve(A, X.T @ y)
        pred = Xe @ w

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(HEADER)
        for i, v in enumerate(pred):
            wr.writerow([i, f"{float(v):.6f}"])
    print(f"wrote {len(pred)} predictions to {out}")


if __name__ == "__main__":
    main()
