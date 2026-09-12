HYPOTHESIS: Expanding feature set with video engagement statistics and user favorite tags via a complete, compact Factorization Machine implementation improves within-user ranking.

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
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tag', 'video_type', 'upload_type', 
          'music_type', 'play_progress', 'user_active', 'follow_range', 'fav_tag', 'fav_author', 'like_rate', 'finish_rate']

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

def load_data(data_dir):
    vid_features = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid_features[r['video_id']] = {
                'author_id': r['author_id'], 'tag': r['tag'], 'video_type': r['video_type'],
                'upload_type': r['upload_type'], 'music_type': r['music_type']
            }
    
    vid_stats = {}
    with open(os.path.join(data_dir, 'video_features_statistic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            s = float(r.get('show_cnt', 0) or 0)
            l = float(r.get('like_cnt', 0) or 0)
            c = float(r.get('complete_play_cnt', 0) or 0)
            p = float(r.get('play_progress', 0.0) or 0)
            vid_stats[r['video_id']] = (p, l/(s+1), c/(s+1))

    user_features = {}
    with open(os.path.join(data_dir, 'user_features_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            user_features[r['user_id']] = {'user_active': r['user_active_degree'], 'follow_range': r['follow_user_num_range']}

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vid, uid = r['video_id'], r['user_id']
                vf = vid_features.get(vid, {k: 'UNK' for k in ['author_id', 'tag', 'video_type', 'upload_type', 'music_type']})
                uf = user_features.get(uid, {'user_active': 'UNK', 'follow_range': 'UNK'})
                vs = vid_stats.get(vid, (0.0, 0.0, 0.0))
                rows.append((int(r['date']), uid, vid, vf['author_id'], r['tab'], float(r['duration_ms']),
                             1 if r[LABEL] != '0' else 0, vf['tag'], vf['video_type'], vf['upload_type'],
                             vf['music_type'], vs[0], uf['user_active'], uf['follow_range'], vs[1], vs[2]))
    
    tr_rows = [x for x in rows if SPLITS['train'][0] <= x[0] <= SPLITS['train'][1]]
    u_tags, u_auths = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    for r in tr_rows:
        if r[6] == 1:
            u_tags[r[1]][r[7]] += 1
            u_auths[r[1]][r[3]] += 1
    user_favs = {u: (max(tags, key=tags.get) if tags else 'UNK', max(u_auths[u], key=u_auths[u].get) if u_auths[u] else 'UNK') for u in u_tags}

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out, user_favs

def encode_data(splits, user_favs):
    tr = splits['train']
    edges_dur = _bucket_edges([x[5] for x in tr])
    edges_prog = _bucket_edges([x[11] for x in tr])
    edges_like = _bucket_edges([x[14] for x in tr])
    edges_fin = _bucket_edges([x[15] for x in tr])

    def raw(x):
        fav = user_favs.get(x[1], ('UNK', 'UNK'))
        return [
            x[1], x[2], x[3], x[4], 
            str(int(np.searchsorted(edges_dur, x[5]))),
            x[7], x[8], x[9], x[10],
            str(int(np.searchsorted(edges_prog, x[11]))),
            x[12], x[13], fav[0], fav[1],
            str(int(np.searchsorted(edges_like, x[14]))),
            str(int(np.searchsorted(edges_fin, x[15])))
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

class CompactFM:
    def __init__(self, dim, k=16, lr=0.001, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0.0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr

    def predict(self, X):
        vx = self.V[X]
        sum_v = np.sum(vx, axis=1)
        sum_v_sq = np.sum(vx ** 2, axis=1)
        factor = 0.5 * np.sum(sum_v ** 2 - sum_v_sq, axis=1)
        linear = np.sum(self.W[X], axis=1)
        logits = linear + factor + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))

    def step(self, X, y):
        pred = self.predict(X)
        err = pred - y
        batch_size = len(X)
        
        # Gradients & updates
        self.b -= self.lr * np.mean(err)
        for j in range(X.shape[1]):
            self.W[X[:, j]] -= self.lr * err
        
        for f in range(X.shape[1]):
            x_f = X[:, f]
            v_f = self.V[x_f]
            vx = self.V[X]
            sum_v = np.sum(vx, axis=1)
            grad_v = err[:, None] * (sum_v - v_f)
            np.add.at(self.V, x_f, -self.lr * grad_v)
            
        return float(np.mean(-(y * np.log(np.maximum(pred, 1e-7)) + (1 - y) * np.log(np.maximum(1 - pred, 1e-7)))))

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

    m = CompactFM(dim, k=16, lr=0.001, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    best, best_state, bad = -1.0, None, 0
    bs = 8192

    for ep in range(1, 15):
        te = time.time()
        idx = rng.permutation(len(ytr))
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"Epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} | {time.time()-te:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 3:
                print(f"Early stop at epoch {ep}")
                break

    if best_state is not None:
        m.V, m.W, m.b = best_state

    rows = splits[a.split]
    if not rows:
        raise SystemExit(f"Split '{a.split}' is empty")
    X = enc[a.split][0]
    write_scores(a.out, rows, m.predict(X))
    print(f"Wrote {a.out} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()