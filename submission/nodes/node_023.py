HYPOTHESIS: Expanding the factorized model with user-favorite tags, video statistics (like rate, finish rate), and robust categorical embeddings will improve ranking performance while keeping the code compact.

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
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tag', 'video_type', 'fav_tag', 'like_bucket']

class FM:
    def __init__(self, p, k=16, lr=0.001, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (p, k)).astype(np.float32)
        self.W = np.zeros(p, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr

    def predict_raw(self, X):
        vx = self.V[X]
        sum_v = vx.sum(axis=1)
        sum_v_sq = (vx ** 2).sum(axis=1)
        inter = 0.5 * (sum_v ** 2 - sum_v_sq).sum(axis=1)
        linear = self.W[X].sum(axis=1)
        return self.b + linear + inter

    def predict(self, X, bs=32768):
        return 1.0 / (1.0 + np.exp(-np.clip(self.predict_raw(X), -15.0, 15.0)))

    def step(self, X, y):
        preds = self.predict(X)
        err = preds - y
        B = len(y)
        
        vx = self.V[X]
        sum_v = vx.sum(axis=1, keepdims=True)
        grad_V = err[:, :, None] * (sum_v - vx)
        
        self.b -= self.lr * err.mean()
        for j in range(X.shape[1]):
            np.add.at(self.W, X[:, j], -self.lr * err)
            np.add.at(self.V, X[:, j], -self.lr * grad_V[:, j, :])
        return float((err ** 2).mean())

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

def load_data(data_dir):
    vid_features = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid_features[r['video_id']] = {'author_id': r['author_id'], 'tag': r['tag'], 'video_type': r['video_type']}
    
    vid_stats = {}
    with open(os.path.join(data_dir, 'video_features_statistic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            s = float(r.get('show_cnt', 0) or 0)
            l = float(r.get('like_cnt', 0) or 0)
            vid_stats[r['video_id']] = l / (s + 1)

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vid, uid = r['video_id'], r['user_id']
                vf = vid_features.get(vid, {'author_id': 'UNK', 'tag': 'UNK', 'video_type': 'UNK'})
                like_r = vid_stats.get(vid, 0.0)
                rows.append((int(r['date']), uid, vid, vf['author_id'], r['tab'], float(r['duration_ms']),
                             1 if r[LABEL] != '0' else 0, vf['tag'], vf['video_type'], like_r))
    
    tr_rows = [x for x in rows if SPLITS['train'][0] <= x[0] <= SPLITS['train'][1]]
    u_tags = defaultdict(lambda: defaultdict(int))
    for r in tr_rows:
        if r[6] == 1:
            u_tags[r[1]][r[7]] += 1
    user_favs = {u: max(tags, key=tags.get) for u, tags in u_tags.items()}

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out, user_favs

def encode_data(splits, user_favs):
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr])
    like_edges = _bucket_edges([x[9] for x in tr])

    def raw(x):
        fav = user_favs.get(x[1], 'UNK')
        dur_b = str(int(np.searchsorted(dur_edges, x[5])))
        like_b = str(int(np.searchsorted(like_edges, x[9])))
        return [x[1], x[2], x[3], x[4], dur_b, x[7], x[8], fav, like_b]

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    splits, user_favs = load_data(a.data_dir)
    enc, dim = encode_data(splits, user_favs)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    m = FM(dim, k=16, lr=0.01, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, 15):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            m.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"Epoch {ep} | Valid primary: {va['primary']:.4f}")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 2:
                break

    m.V, m.W, m.b = best_state
    rows = splits[a.split]
    X = enc[a.split][0]
    
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        scores = m.predict(X)
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])
    print(f"Successfully wrote {len(rows)} rows to {a.out}")

if __name__ == "__main__":
    main()