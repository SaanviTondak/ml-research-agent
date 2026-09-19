"""A deliberately mediocre reference: ridge regression on raw features.

It leaves real headroom - the generator puts an interaction term in the target
that a linear model on raw features cannot represent - so an agent pointed at
this task has something genuine to find rather than only hyperparameters to
wiggle.

--jitter is what makes this task useful for calibration. It injects a
controlled, seed-dependent perturbation into the fitted weights, so the
seed-to-seed spread of the score is a *knob* rather than a property that has
to be discovered. It is deliberately not called --sigma: the resulting spread
is monotone in it but not equal to it (RMSE is quadratic in the error, so the
relationship bends), and tests should take ground-truth sigma from measuring
many seeds rather than from this flag. jitter=0 gives an exactly deterministic
task, which is the degenerate case the policy layer has to survive.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

HEADER = ["row_id", "score"]


def load_split(data_dir, split):
    path = Path(data_dir) / f"{split}.csv"
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        has_target = head[-1] == "target"
        X, y = [], []
        for rec in rd:
            vals = [float(v) for v in rec]
            if has_target:
                X.append(vals[:-1]); y.append(vals[-1])
            else:
                X.append(vals)
    return np.asarray(X), (np.asarray(y) if y else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="seed-dependent weight perturbation; controls (but "
                         "does not equal) the resulting noise floor")
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()

    Xtr, ytr = load_split(a.data_dir, "train")
    Xev, _ = load_split(a.data_dir, a.split)

    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xev = np.hstack([Xev, np.ones((len(Xev), 1))])
    A = Xtr.T @ Xtr + a.alpha * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)

    if a.jitter:
        # Seeded perturbation: the whole point of the task.
        rng = np.random.default_rng(a.seed)
        w = w + rng.normal(scale=a.jitter * 3.0, size=w.shape)

    pred = Xev @ w
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(HEADER)
        for i, v in enumerate(pred):
            wr.writerow([i, f"{v:.6f}"])
    print(f"wrote {len(pred)} predictions to {out}")


if __name__ == "__main__":
    main()
