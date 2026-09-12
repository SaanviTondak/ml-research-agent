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
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tag', 'video_type', 'like_bucket', 'finish_bucket']

class FM:
    def __init__(self, p, k=16, lr=0.001, seed=0):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 0.01, size=(p,)).astype(np.float32)
        self.V = self.rng.normal(0, 0.01, size=(p, k)).astype(np.float32)
        self.b = np.float32(0.0)
        self.lr = lr

    def predict(self, X):
        linear = self.W[X].sum(axis=1) + self.b
        V_x = self.V[X]
        sum_v = V_x.sum(axis=1)
        sum_v_sq = (V_x ** 2).sum(axis=1)
        interaction = 0.5 * ((sum_v ** 2) - sum_v_sq).sum(axis=1)
        return 1.0 / (1.0 + np.exp(-np.clip(linear + interaction, -20.0, 20.0)))

    def step(self, X, y):
        p = self.predict(X)
        err = p - y
        bs = len(y)
        grad_scale = err / bs
        
        self.b -= self.lr * grad_scale.sum()
        for j in range(X.shape[1]):
            xj = X[:, j]
            np.add.at(self.W, xj, grad_scale)
            V_xj = self.V[xj]
            sum_v_j = V_xj.sum(axis=1)
            for k in range(self.V.shape[1]):
                v_grad = grad_scale[:, None] * (sum_v_j[:, k:k+1] - V_xj[:, k])
                np.add.at(self.V[:, k], xj, v_grad.ravel())
        return float(np.mean(-y * np.log(p + 1e-7) - (1 - y) * np.log(1 - p + 1e-7)))

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

def load_data(data_dir):
    vid_features = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid_features[r['video_id']] = {
                'author_id': r['author_id'], 'tag': r['tag'], 'video_type': r['video_type']
            }
    
    vid_stats = {}
    with open(os.path.join(data_dir, 'video_features_statistic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            s = float(r.get('show_cnt', 0) or 0)
            l = float(r.get('like_cnt', 0) or 0)
            c = float(r.get('complete_play_cnt', 0) or 0)
            vid_stats[r['video_id']] = (l / (s + 1), c / (s + 1))

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vid = r['video_id']
                vf = vid_features.get(vid, {'author_id': 'UNK', 'tag': 'UNK', 'video_type': 'UNK'})
                vs = vid_stats.get(vid, (0.0, 0.0))
                rows.append((int(r['date']), r['user_id'], vid, vf['author_id'], r['tab'], 
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0, 
                             vf['tag'], vf['video_type'], vs[0], vs[1]))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def encode_data(splits):
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr])
    like_edges = _bucket_edges([x[9] for x in tr])
    finish_edges = _bucket_edges([x[10] for x in tr])

    def raw(x):
        return [
            x[1], x[2], x[3], x[4],
            str(int(np.searchsorted(dur_edges, x[5]))),
            x[7], x[8],
            str(int(np.searchsorted(like_edges, x[9]))),
            str(int(np.searchsorted(finish_edges, x[10])))
        ]

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
    ap.add_argument("--patience", type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    splits = load_data(a.data_dir)
    print(f"Loaded data in {time.time()-t0:.1f}s")

    enc, dim = encode_data(splits)
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
        print(f"Epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-te:.1f}s")
        
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stop at epoch {ep}")
                break

    if best_state is not None:
        m.V, m.W, m.b = best_state
    print(f"Best valid primary {best:.4f}")

    rows = splits[a.split]
    if not rows:
        raise SystemExit(f"Split '{a.split}' is empty")
    X = enc[a.split][0]
    write_scores(a.out, rows, m.predict(X))
    print(f"Wrote {a.out} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()