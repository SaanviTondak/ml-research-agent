HYPOTHESIS: Expanding feature fields with video statistics and user-video interaction preference helps the Factorization Machine learn better representations without exceeding token length limits.

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
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tag', 'video_type', 'fav_tag', 'fav_author', 'like_rate']

class FM:
    def __init__(self, dim, k=16, lr=0.001, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr

    def predict(self, X):
        vx = self.V[X]
        sum_v = vx.sum(axis=1)
        sum_v_sq = (vx ** 2).sum(axis=1)
        inter = 0.5 * (sum_v ** 2 - sum_v_sq).sum(axis=1)
        linear = self.W[X].sum(axis=1)
        return 1.0 / (1.0 + np.exp(-np.clip(self.b + linear + inter, -20, 20)))

    def step(self, X, y):
        preds = self.predict(X)
        err = preds - y
        batch_size = len(X)
        
        # Gradients
        self.b -= self.lr * err.mean()
        
        # Linear grads
        for i in range(X.shape[1]):
            np.add.at(self.W, X[:, i], err)
        self.W -= self.lr * (err[:, None] * (X != -1)).sum(axis=0) / batch_size # simple approx

        # V grads (simplified SGD)
        vx = self.V[X]
        sum_v = vx.sum(axis=1, keepdims=True)
        for f in range(X.shape[1]):
            feat_idx = X[:, f]
            valid = feat_idx != -1
            if not valid.any(): continue
            grad_v = err[:, None, :] * (sum_v - vx[:, f:f+1, :])
            for i in range(len(X)):
                if valid[i]:
                    self.V[feat_idx[i]] -= self.lr * grad_v[i, 0]
        
        return float(np.mean(-y * np.log(preds + 1e-7) - (1 - y) * np.log(1 - preds + 1e-7)))

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

def load_and_encode(data_dir):
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
                rate = vid_stats.get(vid, 0.0)
                rows.append((int(r['date']), uid, vid, vf['author_id'], r['tab'], float(r['duration_ms']),
                             1 if r[LABEL] != '0' else 0, vf['tag'], vf['video_type'], rate))
    
    tr_rows = [x for x in rows if SPLITS['train'][0] <= x[0] <= SPLITS['train'][1]]
    durations = [x[5] for x in tr_rows]
    rates = [x[9] for x in tr_rows]
    dur_edges = _bucket_edges(durations, 10)
    rate_edges = _bucket_edges(rates, 10)

    u_tags, u_auths = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    for r in tr_rows:
        if r[6] == 1:
            u_tags[r[1]][r[7]] += 1
            u_auths[r[1]][r[3]] += 1
    user_favs = {u: (max(tags, key=tags.get), max(u_auths[u], key=u_auths[u].get)) for u, tags in u_tags.items()}

    def raw(x):
        fav = user_favs.get(x[1], ('UNK', 'UNK'))
        d_b = str(int(np.searchsorted(dur_edges, x[5])))
        r_b = str(int(np.searchsorted(rate_edges, x[9])))
        return [x[1], x[2], x[3], x[4], d_b, x[7], x[8], fav[0], fav[1], r_b]

    vocabs = [dict() for _ in FIELDS]
    for x in tr_rows:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])

    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    out = {}
    for name, (lo, hi) in SPLITS.items():
        rws = [x for x in rows if lo <= x[0] <= hi]
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        out[name] = (X, y, users, rws)

    return out, int(sum(field_dims))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    splits, dim = load_and_encode(a.data_dir)
    Xtr, ytr, _, _ = splits["train"]
    Xva, yva, uva, _ = splits["valid"]

    m = FM(dim, k=16, lr=0.005, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, 15):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            m.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"epoch {ep:2d} | valid primary {va['primary']:.4f} (GAUC {va['GAUC']:.4f})")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 3: break

    m.V, m.W, m.b = best_state
    _, _, _, rows = splits[a.split]
    X = splits[a.split][0]
    
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (x, s) in enumerate(zip(rows, m.predict(X))):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()