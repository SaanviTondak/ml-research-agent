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
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tag', 'fav_tag', 'like_bucket']

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

def load_data(data_dir):
    vid_features = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid_features[r['video_id']] = {'author_id': r['author_id'], 'tag': r['tag']}
    
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
                vf = vid_features.get(vid, {'author_id': 'UNK', 'tag': 'UNK'})
                vs = vid_stats.get(vid, 0.0)
                rows.append((int(r['date']), uid, vid, vf['author_id'], r['tab'], float(r['duration_ms']),
                             1 if r[LABEL] != '0' else 0, vf['tag'], vs))
    
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
    edges_dur = _bucket_edges([x[5] for x in tr])
    edges_like = _bucket_edges([x[8] for x in tr])

    def raw(x):
        fav = user_favs.get(x[1], 'UNK')
        dur_b = str(int(np.searchsorted(edges_dur, x[5])))
        like_b = str(int(np.searchsorted(edges_like, x[8])))
        return [x[1], x[2], x[3], x[4], dur_b, x[7], fav, like_b]

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
        self.V = rng.normal(0.0, 0.01, (p, k)).astype(np.float32)
        self.W = np.zeros(p, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr

    def predict(self, X):
        vx = self.W[X].sum(axis=1) + self.b
        v_emb = self.V[X]
        sum_emb = v_emb.sum(axis=1)
        sum_sq_emb = (v_emb ** 2).sum(axis=1)
        inter = 0.5 * (sum_emb ** 2 - sum_sq_emb).sum(axis=1)
        return 1.0 / (1.0 + np.exp(-np.clip(vx + inter, -20.0, 20.0)))

    def step(self, X, y):
        preds = self.predict(X)
        grad = preds - y
        
        self.b -= self.lr * grad.mean()
        np.add.at(self.W, X, -self.lr * grad[:, None])
        
        v_emb = self.V[X]
        sum_emb = v_emb.sum(axis=1, keepdims=True)
        for i in range(X.shape[1]):
            xi = X[:, i]
            vi = v_emb[:, i]
            g_v = grad[:, None] * (sum_emb[:, 0] - vi)
            np.add.at(self.V, xi, -self.lr * g_v)
        
        loss = -np.mean(y * np.log(preds + 1e-7) + (1 - y) * np.log(1 - preds + 1e-7))
        return loss

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
    a = ap.parse_args()

    t0 = time.time()
    splits, user_favs = load_data(a.data_dir)
    print(f"Loaded data in {time.time()-t0:.1f}s")

    enc, dim = encode_data(splits, user_favs)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    m = FM(dim, k=16, lr=0.001, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, 25):
        te = time.time()
        idx = rng.permutation(len(ytr))
        bs = 8192
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"Epoch {ep:2d} | Loss {np.mean(losses):.4f} | Valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} Primary {va['primary']:.4f} | {time.time()-te:.1f}s")
        
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 3:
                print(f"Early stop at epoch {ep}")
                break

    m.V, m.W, m.b = best_state
    print(f"Best valid primary {best:.4f}")

    rows = splits[a.split]
    if not rows:
        raise SystemExit(f"Split '{a.split}' is empty")
    X = enc[a.split][0]
    write_scores(a.out, rows, m.predict(X))
    print(f"Wrote {a.out} in {time.time()-t0:.1f}s total")

if __name__ == "__main__":
    main()