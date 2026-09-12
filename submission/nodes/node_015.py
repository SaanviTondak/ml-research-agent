import argparse
import csv
import os
import time
import numpy as np
from collections import defaultdict
import data
from evaluate import evaluate

HEADER = ["row_id", "user_id", "video_id", "score"]
FIELDS = ['user_id', 'video_id', 'author_id', 'tag', 'tab', 'dur_bucket', 'fav_tag']

class FM:
    def __init__(self, dim, k=16, lr=0.001, seed=0):
        self.k, self.lr = k, lr
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.normal(0, 0.01, dim).astype(np.float32)
        self.v = self.rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.b = 0.0

    def _logits(self, X):
        v_x = self.v[X]
        v_sum = v_x.sum(axis=1)
        v_sum_sq = (v_x**2).sum(axis=1)
        interaction = 0.5 * (v_sum**2 - v_sum_sq).sum(axis=1)
        return self.w[X].sum(axis=1) + self.b + interaction

    def predict(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(self._logits(X), -20, 20)))

    def step(self, X, y):
        B = X.shape[0]
        v_x = self.v[X]
        v_sum = v_x.sum(axis=1)
        v_sum_sq = (v_x**2).sum(axis=1)
        logits = self.w[X].sum(axis=1) + self.b + 0.5 * (v_sum**2 - v_sum_sq).sum(axis=1)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        grad = probs - y
        self.b -= self.lr * grad.mean()
        for i in range(X.shape[1]):
            np.add.at(self.w, X[:, i], -self.lr * grad / B)
            gv = grad[:, np.newaxis] * (v_sum - v_x[:, i, :])
            np.add.at(self.v, X[:, i], -self.lr * gv / B)
        return -np.mean(y * np.log(probs + 1e-9) + (1-y) * np.log(1-probs + 1e-9))

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
    splits = data.load(a.data_dir)
    vid2tag = {}
    with open(os.path.join(a.data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh): vid2tag[r['video_id']] = r['tag']

    user_tag_counts = defaultdict(lambda: defaultdict(int))
    for x in splits['train']:
        if x[6] == 1: user_tag_counts[x[1]][vid2tag.get(x[2], 'UNK')] += 1
    user2fav = {u: max(tags, key=tags.get) for u, tags in user_tag_counts.items()}

    tr_durs = [x[5] for x in splits['train']]
    edges = np.quantile(np.asarray(tr_durs), np.linspace(0, 1, 11)[1:-1])

    def raw(x):
        uid, vid, aid, tab, dur = x[1], x[2], x[3], x[4], x[5]
        return [uid, vid, aid, vid2tag.get(vid, 'UNK'), tab, str(int(np.searchsorted(edges, dur))), user2fav.get(uid, 'UNK')]

    vocabs = [{} for _ in range(len(FIELDS))]
    for x in splits['train']:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]: vocabs[i][v] = len(vocabs[i])
    
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    dim = int(sum(field_dims))

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, len(vocabs[i])) + offsets[i]
            y[n] = x[6]
        enc[name] = (X, y, [x[1] for x in rws])

    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    m = FM(dim, k=a.k, lr=a.lr, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, a.epochs + 1):
        idx = rng.permutation(len(ytr))
        losses = [m.step(Xtr[idx[i:i+a.bs]], ytr[idx[i:i+a.bs]]) for i in range(0, len(idx), a.bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"Epoch {ep:2d} | Loss {np.mean(losses):.4f} | GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f}")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.v.copy(), m.w.copy(), m.b)
        else:
            bad += 1
            if bad >= a.patience: break

    m.v, m.w, m.b = best_state
    rows = splits[a.split]
    X_split = enc[a.split][0]
    scores = m.predict(X_split)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])
    print(f"Best primary {best:.4f}. Wrote {a.out} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()