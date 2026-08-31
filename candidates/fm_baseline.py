"""Candidate 0 - the organizer's Factorization Machine, in the candidate contract.

This is the reference implementation of the contract every agent-written
candidate must satisfy:

    python3 <script> --data_dir DIR --split {train,valid,test} --out FILE

  * reads ONLY from --data_dir (never a hard-coded path to the full dataset);
  * trains on the train split and selects on valid;
  * writes `row_id,user_id,video_id,score` for --split, one line per row of
    data.load(data_dir)[split], in that exact order;
  * prints whatever it likes to stdout - the loop captures it as the trace.

Why the script takes --split rather than always writing valid: at the end of
the project the winning candidate has to be re-run against the test split by
seal/final_score.py. Making that a parameter now means the winner needs no
edits later, and the agent never gets to pass anything but 'valid'.

The model itself is imported unmodified from the organizer's baseline.py, so
this candidate reproduces the official baseline exactly rather than
approximating it. Its purpose is to prove the harness, not to score well.
"""
import argparse
import csv
import time

import numpy as np

from data import load, encode          # organizer's, via PYTHONPATH
from evaluate import evaluate
from baseline import FM                # organizer's FM, unmodified

HEADER = ["row_id", "user_id", "video_id", "score"]


def write_scores(path, rows, scores):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--patience", type=int, default=4)
    a = ap.parse_args()

    t0 = time.time()
    splits = load(a.data_dir)
    print(f"loaded {a.data_dir}: "
          + ", ".join(f"{k}={len(v):,d}" for k, v in splits.items())
          + f"  ({time.time()-t0:.1f}s)")

    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    m = FM(dim, k=a.k, lr=a.lr, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, a.epochs + 1):
        te = time.time()
        idx = rng.permutation(len(ytr))
        losses = [m.step(Xtr[idx[i:i + a.bs]], ytr[idx[i:i + a.bs]])
                  for i in range(0, len(idx), a.bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} "
              f"| valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
              f"primary {va['primary']:.4f} | {time.time()-te:.1f}s")
        # Rule 2: keep the validation-best checkpoint, not the last one.
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop at epoch {ep}")
                break

    m.V, m.W, m.b = best_state
    print(f"  best valid primary {best:.4f}")

    rows = splits[a.split]
    if not rows:
        raise SystemExit(
            f"split '{a.split}' is empty in {a.data_dir} - nothing to score")
    X = enc[a.split][0]
    write_scores(a.out, rows, m.predict(X))
    print(f"wrote {a.out}: {len(rows):,d} rows (split={a.split}) "
          f"in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
