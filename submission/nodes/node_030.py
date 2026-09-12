HYPOTHESIS: Expanding the Factorization Machine with video tags, play progress, and simple user-favorite features can improve ranking performance, implemented compactly to fit the length limits.

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
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tag', 'video_type', 'play_prog', 'fav_tag', 'fav_auth']

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

class FM:
    def __init__(self, p, k=16, lr=0.001, seed=0):
        self.rng = np.random.default_rng(seed)
        self.v = self.rng.normal(0, 0.01, size=(p, k)).astype(np.float32)
        self.w = np.zeros(p, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr

    def predict_raw(self, X):
        linear = self.w[X].sum(axis=1) + self.b
        v_embed = self.v[X]  # (N, F, k)
        sum_v = v_embed.sum(axis=1)  # (N, k)
        sum_v_sq = (v_embed ** 2).sum(axis=1)
        interaction = 0.5 * ((sum_v ** 2) - sum_v_sq).sum(axis=1)
        return linear + interaction

    def predict(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(self.predict_raw(X), -15.0, 15.0)))

    def step(self, X, y):
        preds = self.predict(X)
        err = preds - y  # (N,)
        
        # SGD updates
        self.b -= self.lr * err.mean()
        for i in range(X.shape[0]):
            feats = X[i]
            e = err[i]
            self.w[feats] -= self.lr * e
            v_feats = self.v[feats] # (F, k)
            sum_v = v_feats.sum(axis=0) # (k,)
            for f_idx, feat in enumerate(feats):
                grad_v = e * (sum_v - v_feats[f_idx])
                self.v[feat] -= self.lr * grad_v
        return float(np.mean(err ** 2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16384)
    ap.add_argument("--patience", type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    
    vid_features = {}
    with open(os.path.join(a.data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid_features[r['video_id']] = {'author_id': r['author_id'], 'tag': r['tag'], 'video_type': r['video_type']}
            
    vid_stats = {}
    with open(os.path.join(a.data_dir, 'video_features_statistic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid_stats[r['video_id']] = float(r.get('play_progress', 0.0) or 0)

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(a.data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vid, uid = r['video_id'], r['user_id']
                vf = vid_features.get(vid, {'author_id': 'UNK', 'tag': 'UNK', 'video_type': 'UNK'})
                prog = vid_stats.get(vid, 0.0)
                rows.append((int(r['date']), uid, vid, vf['author_id'], r['tab'], float(r['duration_ms']),
                             1 if r[LABEL] != '0' else 0, vf['tag'], vf['video_type'], prog))

    tr_rows = [x for x in rows if SPLITS['train'][0] <= x[0] <= SPLITS['train'][1]]
    u_tags, u_auths = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    for r in tr_rows:
        if r[6] == 1:
            u_tags[r[1]][r[7]] += 1
            u_auths[r[1]][r[3]] += 1
    user_favs = {u: (max(tags, key=tags.get), max(u_auths[u], key=u_auths[u].get)) for u, tags in u_tags.items()}

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]

    edges_dur = _bucket_edges([x[5] for x in tr_rows])
    edges_prog = _bucket_edges([x[9] for x in tr_rows])

    def raw(x):
        fav = user_favs.get(x[1], ('UNK', 'UNK'))
        return [
            x[1], x[2], x[3], x[4], 
            str(int(np.searchsorted(edges_dur, x[5]))),
            x[7], x[8], 
            str(int(np.searchsorted(edges_prog, x[9]))),
            fav[0], fav[1]
        ]

    vocabs = [dict() for _ in FIELDS]
    for x in tr_rows:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    total_dim = int(sum(field_dims))

    enc = {}
    for name, rws in out.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)

    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    m = FM(total_dim, k=a.k, lr=a.lr, seed=a.seed)
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
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.v.copy(), m.w.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop at epoch {ep}")
                break

    if best_state is not None:
        m.v, m.w, m.b = best_state
    print(f"  best valid primary {best:.4f}")

    rows_target = out[a.split]
    if not rows_target:
        raise SystemExit(f"split '{a.split}' is empty")
    X_target = enc[a.split][0]
    scores = m.predict(X_target)

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (x, s) in enumerate(zip(rows_target, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])
    print(f"wrote {a.out}: {len(rows_target):,d} rows in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()