HYPOTHESIS: Streamlining the script and incorporating user favorite tag/author features with simple interaction history boosts within-user ranking without truncation or complexity issues.

import argparse
import csv
import os
import time
from collections import defaultdict
import numpy as np

from evaluate import evaluate

HEADER = ["row_id", "user_id", "video_id", "score"]
LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421), 'valid': (20220422, 20220428), 'test': (20220429, 20220508)}
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tag', 'fav_tag']

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

def load_data(data_dir):
    vid_features = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid_features[r['video_id']] = {'author_id': r['author_id'], 'tag': r['tag']}

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vid, uid = r['video_id'], r['user_id']
                vf = vid_features.get(vid, {'author_id': 'UNK', 'tag': 'UNK'})
                rows.append((int(r['date']), uid, vid, vf['author_id'], r['tab'], float(r['duration_ms']),
                             1 if r[LABEL] != '0' else 0, vf['tag']))
    
    tr_rows = [x for x in rows if SPLITS['train'][0] <= x[0] <= SPLITS['train'][1]]
    u_tags = defaultdict(lambda: defaultdict(int))
    for r in tr_rows:
        if r[6] == 1:
            u_tags[r[1]][r[7]] += 1
    user_favs = {u: max(tags, key=tags.get) if tags else 'UNK' for u, tags in u_tags.items()}

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out, user_favs

def encode_data(splits, user_favs):
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        fav = user_favs.get(x[1], 'UNK')
        dur = str(int(np.searchsorted(edges, x[5])))
        return [x[1], x[2], x[3], x[4], dur, x[7], fav]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))

class FM:
    def __init__(self, p, k=16, lr=0.001, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, size=(p, k)).astype(np.float32)
        self.W = np.zeros(p, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr

    def predict(self, X):
        linear = self.W[X].sum(axis=1) + self.b
        v_X = self.V[X]
        sum_v = v_X.sum(axis=1)
        sum_v_sq = (v_X ** 2).sum(axis=1)
        factor = 0.5 * ((sum_v ** 2) - sum_v_sq).sum(axis=1)
        return 1.0 / (1.0 + np.exp(-np.clip(linear + factor, -20.0, 20.0)))

    def step(self, X, y):
        pred = self.predict(X)
        err = pred - y
        
        self.b -= self.lr * err.mean()
        for j in range(X.shape[1]):
            xj = X[:, j]
            self.W[xj] -= self.lr * (err * 1.0).dot(np.ones_like(xj, dtype=np.float32)) / len(X)
            
        v_X = self.V[X]
        sum_v = v_X.sum(axis=1, keepdims=True)
        for f in range(X.shape[1]):
            x_f = X[:, f]
            grad_V = (err[:, None, None] * (sum_v - v_X[:, f:f+1])).mean(axis=0)
            self.V[x_f] -= self.lr * grad_V
        return float(np.mean(-y * np.log(np.clip(pred, 1e-7, 1)) - (1 - y) * np.log(np.clip(1 - pred, 1e-7, 1))))

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
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8192)
    a = ap.parse_args()

    t0 = time.time()
    splits, user_favs = load_data(a.data_dir)
    print(f"loaded data in {time.time()-t0:.1f}s")

    enc, dim = encode_data(splits, user_favs)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    m = FM(dim, k=a.k, lr=a.lr, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    best, best_state = -1.0, None

    for ep in range(1, a.epochs + 1):
        idx = rng.permutation(len(ytr))
        losses = [m.step(Xtr[idx[i:i + a.bs]], ytr[idx[i:i + a.bs]]) for i in range(0, len(idx), a.bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f}")
        if va["primary"] > best:
            best = va["primary"]
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))

    m.V, m.W, m.b = best_state
    rows = splits[a.split]
    X = enc[a.split][0]
    write_scores(a.out, rows, m.predict(X))
    print(f"wrote {a.out}: {len(rows):,d} rows")

if __name__ == "__main__":
    main()