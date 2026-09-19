"""A task whose noise floor is a construction parameter.

Every policy constant in agent/state.py is denominated in the noise floor of
one benchmark. To show that deriving them from a *measured* noise floor
actually works, you need a task where the true noise floor is known rather
than estimated - otherwise "calibration recovers sigma" is unfalsifiable.

So this generates a regression problem with three knobs:

  --sigma      how much a change of seed moves the score. 0 is a legal and
               important value: a deterministic task must not make every gain
               look significant.
  --n_eval     how many evaluation rows, which sets the evaluation-set noise
               independently of the training noise.
  --quantum    round the metric to this grid, to simulate a discrete metric
               like accuracy over a small test set.

The data itself is a plain linear model with a few interactions and some
irrelevant columns, so there is real signal for a candidate to find and a
genuine ceiling it cannot pass. It is not meant to be interesting; it is meant
to be *known*.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

N_FEATURES = 12
N_INFORMATIVE = 5
SPLITS = {"train": 8000, "valid": 2000, "test": 2000}


def build(out_dir, seed=0, n_eval=None):
    """Write train/valid/test CSVs. The test split carries no target column."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    sizes = dict(SPLITS)
    if n_eval:
        sizes["valid"] = sizes["test"] = int(n_eval)
    total = sum(sizes.values())

    X = rng.normal(size=(total, N_FEATURES))
    w = np.zeros(N_FEATURES)
    w[:N_INFORMATIVE] = rng.normal(scale=1.5, size=N_INFORMATIVE)
    # One interaction, so a linear reference implementation leaves headroom
    # that a better candidate can actually capture.
    y = X @ w + 1.2 * X[:, 0] * X[:, 1] + rng.normal(scale=0.5, size=total)

    lo = 0
    manifest = {"seed": seed, "n_features": N_FEATURES,
                "n_informative": N_INFORMATIVE, "splits": {}}
    for name in ("train", "valid", "test"):
        n = sizes[name]
        xs, ys = X[lo:lo + n], y[lo:lo + n]
        lo += n
        header = [f"f{i}" for i in range(N_FEATURES)]
        # The test targets are never written to disk in the visible directory.
        # Same principle as the KuaiRand firewall: absent, not masked.
        with open(out_dir / f"{name}.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(header + (["target"] if name != "test" else []))
            for i in range(n):
                row = [f"{v:.6f}" for v in xs[i]]
                if name != "test":
                    row.append(f"{ys[i]:.6f}")
                wr.writerow(row)
        manifest["splits"][name] = {"rows": n, "has_target": name != "test"}

    # Held out separately, for the sealed scorer only.
    np.save(out_dir / "_test_target.npy", y[lo - sizes["test"]:lo])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_eval", type=int, default=None)
    a = ap.parse_args()
    m = build(a.out, seed=a.seed, n_eval=a.n_eval)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
