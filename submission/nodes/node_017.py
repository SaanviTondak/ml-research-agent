HYPOTHESIS: Within-user ranking can be improved by adding a few key video statistic features (like play progress and like rate) to the Factorization Machine baseline while keeping the model compact and fully self-contained.

import argparse
import csv
import os
import time
import numpy as np
from evaluate import evaluate

HEADER = ["row_id", "user_id", "video_id", "score"]
LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tag', 'play_progress_bucket', 'like_rate_bucket']

class FM:
    def __init__(self, dim, k=16, lr=0.001, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr

    def predict(self, X):
        linear = self.W[X].sum(axis=1) + self.b
        vx = self.V[X]
        interaction = 0.5 * ((vx.sum(axis=1) ** 2) - (vx ** 2).sum(axis=1)).sum(axis=1)
        z = linear + interaction
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))

    def step(self, X, y):
        pred = self.predict(X)
        grad = pred - y
        loss = -np.mean(y * np.log(np.clip(pred, 1e-7, 1.0)) + (1 - y) * np.log(np.clip(1 - pred, 1e-7, 1.0)))

        batch_size = len(X)
        g = grad / batch_size

        self.b -= self.lr * g.sum()
        self.W[X] -= self.lr * g[:, None]

        vx = self.V[X]
        sum_vx = vx.sum(axis=1, keepdims=True)
        for f_idx in range(X.shape[1]):
            feats = X[:, f_idx]
            g_f = g[:, None]
            v_f = self.V[feats]
            grad_v = g_f * (sum_vx[:, 0, :] - v_f)
            self.V[feats] -= self.lr * grad_v

        return loss

def load_data(data_dir):
    vid2info = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2info[r['video_id']] = (r['author_id'], r.get('tag', 'UNK'))

    vid2stats = {}
    with open(os.path.join(data_dir, 'video_features_statistic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            show = float(r.get('show_cnt', 1) or 1)
            like = float(r.get('like_cnt', 0) or 0)
            prog = float(r.get('play_progress', 0.0) or 0)
            vid2stats[r['video_id']] = (prog, like / (show + 1.0))

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vid = r['video_id']
                author, tag = vid2info.get(vid, ('UNK', 'UNK'))
                prog, like_rate = vid2stats.get(vid, (0.0, 0.0))
                rows.append((
                    int(r['date']), r['user_id'], vid,
                    author, r['tab'], float(r['duration_ms']),
                    1 if r[LABEL] != '0' else 0, tag, prog, like_rate
                ))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr])
    prog_edges = _bucket_edges([x[8] for x in tr])
    rate_edges = _bucket_edges([x[9] for x in tr])

    def raw(x):
        return [
            x[1],                                           # user_id
            x[2],                                           # video_id
            x[3],                                           # author_id
            x[4],                                           # tab
            str(int(np.searchsorted(dur_edges, x[5]))),    # dur_bucket
            x[7],                                           # tag
            str(int(np.searchsorted(prog_edges, x[8]))),   # prog_bucket
            str(int(np.searchsorted(rate_edges, x[9]))),   # rate_bucket
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
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--patience", type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    splits = load_data(a.data_dir)
    print(f"Loaded data in {time.time()-t0:.1f}s")

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
        raise SystemExit(f"Split '{a.split}' is empty.")
    X = enc[a.split][0]
    write_scores(a.out, rows, m.predict(X))
    print(f"Wrote {a.out} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()