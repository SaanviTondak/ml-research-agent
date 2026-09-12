import argparse
import csv
import time
from collections import defaultdict
import numpy as np

from data import load, encode
from evaluate import evaluate

HEADER = ["row_id", "user_id", "video_id", "score"]


def write_scores(path, rows, scores):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])


class ListwiseFM:
    def __init__(self, num_features, k=16, lr=0.005, weight_decay=1e-5, seed=0):
        rng = np.random.default_rng(seed)
        self.num_features = num_features
        self.k = k
        self.lr = lr
        self.wd = weight_decay

        self.V = rng.normal(0.0, 0.01, size=(num_features, k)).astype(np.float32)
        self.W = np.zeros(num_features, dtype=np.float32)
        self.b = np.float32(0.0)

        # Adam moments
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.float32(0.0)
        self.vb = np.float32(0.0)
        self.t = 0

    def predict(self, X):
        V_X = self.V[X]
        sum_V = np.sum(V_X, axis=1)
        sum_sq_V = np.sum(V_X ** 2, axis=1)
        inter = 0.5 * np.sum(sum_V ** 2 - sum_sq_V, axis=1)
        lin = np.sum(self.W[X], axis=1) + self.b
        return lin + inter

    def _forward(self, X):
        V_X = self.V[X]
        sum_V = np.sum(V_X, axis=1)
        sum_sq_V = np.sum(V_X ** 2, axis=1)
        inter = 0.5 * np.sum(sum_V ** 2 - sum_sq_V, axis=1)
        lin = np.sum(self.W[X], axis=1) + self.b
        return lin + inter, V_X, sum_V

    def step_listwise(self, X_pos, X_neg):
        # X_pos: (B, F)
        # X_neg: (B, C, F)
        B = len(X_pos)
        C = X_neg.shape[1]

        X_neg_flat = X_neg.reshape(-1, X_neg.shape[-1])
        X_combined = np.concatenate([X_pos, X_neg_flat], axis=0)

        s_combined, V_combined, sum_V_combined = self._forward(X_combined)

        s_pos = s_combined[:B]
        s_neg = s_combined[B:].reshape(B, C)

        scores = np.column_stack([s_pos, s_neg])  # (B, C+1)

        # Softmax over the C+1 items
        scores_max = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        loss = -np.mean(np.log(probs[:, 0] + 1e-15))

        # Gradients wrt scores
        grad_scores = probs.copy()
        grad_scores[:, 0] -= 1.0
        grad_scores /= B  # mean gradient over the batch

        grad_spos = grad_scores[:, 0]
        grad_sneg_flat = grad_scores[:, 1:].ravel()
        grad_combined = np.concatenate([grad_spos, grad_sneg_flat], axis=0)

        dV = np.zeros_like(self.V)
        dW = np.zeros_like(self.W)
        db = np.float32(np.sum(grad_combined))

        # Backward pass
        np.add.at(dW, X_combined.ravel(), np.repeat(grad_combined, X_combined.shape[1]))
        grad_V = grad_combined[:, None, None] * (sum_V_combined[:, None, :] - V_combined)
        np.add.at(dV, X_combined.ravel(), grad_V.reshape(-1, self.k))

        # Adam update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr_t = self.lr * np.sqrt(1.0 - beta2 ** self.t) / (1.0 - beta1 ** self.t)

        # Apply weight decay
        if self.wd > 0:
            dV += self.wd * self.V
            dW += self.wd * self.W

        self.mV = beta1 * self.mV + (1.0 - beta1) * dV
        self.vV = beta2 * self.vV + (1.0 - beta2) * (dV ** 2)
        self.V -= lr_t * self.mV / (np.sqrt(self.vV) + eps)

        self.mW = beta1 * self.mW + (1.0 - beta1) * dW
        self.vW = beta2 * self.vW + (1.0 - beta2) * (dW ** 2)
        self.W -= lr_t * self.mW / (np.sqrt(self.vW) + eps)

        self.mb = beta1 * self.mb + (1.0 - beta1) * db
        self.vb = beta2 * self.vb + (1.0 - beta2) * (db ** 2)
        self.b -= lr_t * self.mb / (np.sqrt(self.vb) + eps)

        return float(loss)


def prepare_list_sampler(ytr, users, rng, C=15):
    user_neg = defaultdict(list)
    for idx, (y, u) in enumerate(zip(ytr, users)):
        if y == 0:
            user_neg[u].append(idx)

    all_pos = np.where(ytr == 1)[0]

    # Keep only positive samples where the user has at least one negative sample
    valid_pos = []
    pos_to_neg_pool = []
    for pos_idx in all_pos:
        u = users[pos_idx]
        negs = user_neg[u]
        if len(negs) > 0:
            valid_pos.append(pos_idx)
            pos_to_neg_pool.append(np.array(negs, dtype=np.int32))

    valid_pos = np.array(valid_pos, dtype=np.int32)

    def sample():
        n_pos = len(valid_pos)
        sampled_neg = np.empty((n_pos, C), dtype=np.int32)
        for i, pool in enumerate(pos_to_neg_pool):
            sampled_neg[i] = pool[rng.integers(0, len(pool), size=C)]
        perm = rng.permutation(n_pos)
        return valid_pos[perm], sampled_neg[perm]

    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--neg_samples", type=int, default=15)
    ap.add_argument("--patience", type=int, default=6)
    a = ap.parse_args()

    t0 = time.time()
    splits = load(a.data_dir)
    print(f"loaded {a.data_dir}: "
          + ", ".join(f"{k}={len(v):,d}" for k, v in splits.items())
          + f"  ({time.time()-t0:.1f}s)")

    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]

    rng = np.random.default_rng(a.seed)
    sampler = prepare_list_sampler(ytr, utr, rng, C=a.neg_samples)

    m = ListwiseFM(dim, k=a.k, lr=a.lr, weight_decay=a.wd, seed=a.seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, a.epochs + 1):
        te = time.time()
        pos_idxs, neg_idxs = sampler()
        losses = []
        for i in range(0, len(pos_idxs), a.bs):
            b_pos = pos_idxs[i:i + a.bs]
            b_neg = neg_idxs[i:i + a.bs]
            loss = m.step_listwise(Xtr[b_pos], Xtr[b_neg])
            losses.append(loss)

        va = evaluate(uva, yva, m.predict(Xva))
        print(f"  epoch {ep:2d} | Listwise loss {np.mean(losses):.4f} "
              f"| valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
              f"primary {va['primary']:.4f} | {time.time()-te:.1f}s")

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