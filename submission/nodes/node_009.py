import argparse
import csv
import os
import time
import collections
import numpy as np

from evaluate import evaluate

# Feature set including both standard and higher-quality static attributes
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 
          'video_type', 'upload_type', 'music_type', 'user_active_degree']
SPLITS = {'train': (20220408, 20220421), 'valid': (20220422, 20220428)}

def load_data(data_dir):
    vid_info = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as f:
        for r in csv.DictReader(f):
            vid_info[r['video_id']] = (r['author_id'], r['video_type'], r['upload_type'], r['music_type'])
    
    user_info = {}
    with open(os.path.join(data_dir, 'user_features_pure.csv')) as f:
        for r in csv.DictReader(f):
            user_info[r['user_id']] = r['user_active_degree']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vi = vid_info.get(r['video_id'], ('UNK', 'UNK', 'UNK', 'UNK'))
                ui = user_info.get(r['user_id'], 'UNK')
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vi[0], r['tab'], float(r['duration_ms']),
                             1 if r['long_view'] != '0' else 0,
                             1 if r['is_click'] != '0' else 0,
                             vi[1], vi[2], vi[3], ui))
    
    out = {name: [x for x in rows if lo <= x[0] <= hi] for name, (lo, hi) in SPLITS.items()}
    return out

def encode_data(splits):
    tr = splits['train']
    tr_durs = [x[5] for x in tr]
    edges = np.quantile(np.array(tr_durs), np.linspace(0, 1, 11)[1:-1])

    def get_raw(rws):
        durs = np.array([x[5] for x in rws])
        buckets = np.searchsorted(edges, durs).astype(str)
        return [[x[1], x[2], x[3], x[4], buckets[i], x[8], x[9], x[10], x[11]] for i, x in enumerate(rws)]

    vocabs = [{} for _ in FIELDS]
    for row in get_raw(tr):
        for i, v in enumerate(row):
            if v not in vocabs[i]: vocabs[i][v] = len(vocabs[i])
    
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    
    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty((len(rws), 2), dtype=np.float32)
        users = [x[1] for x in rws]
        for n, row_feat in enumerate(get_raw(rws)):
            for i, v in enumerate(row_feat):
                X[n, i] = vocabs[i].get(v, len(vocabs[i])) + offsets[i]
            y[n, 0], y[n, 1] = rws[n][6], rws[n][7]
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))

class MTL_FM:
    def __init__(self, dim, k=16, lr=0.05, alpha=0.7, seed=0):
        self.k, self.lr, self.alpha = k, lr, alpha
        rng = np.random.default_rng(seed)
        self.V = rng.standard_normal((dim, k)).astype(np.float32) * 0.01
        self.W = np.zeros((dim, 2), dtype=np.float32)
        self.b = np.zeros(2, dtype=np.float32)
        self.num_fields = len(FIELDS)

    def step(self, X, y):
        # Forward
        V_X = self.V[X]
        V_sum = np.sum(V_X, axis=1)
        interact = 0.5 * np.sum(V_sum**2 - np.sum(V_X**2, axis=1), axis=1)
        
        z_long = self.b[0] + np.sum(self.W[X, :, 0], axis=1) + interact
        p_long = 1.0 / (1.0 + np.exp(-np.clip(z_long, -15, 15)))
        
        z_click = self.b[1] + np.sum(self.W[X, :, 1], axis=1) + interact
        p_click = 1.0 / (1.0 + np.exp(-np.clip(z_click, -15, 15)))
        
        # Gradients
        g_long = p_long - y[:, 0]
        g_click = p_click - y[:, 1]
        
        # Update b
        self.b[0] -= self.lr * np.mean(g_long)
        self.b[1] -= self.lr * self.alpha * np.mean(g_click)
        
        # Update W
        X_flat = X.ravel()
        np.add.at(self.W[:, 0], X_flat, -self.lr * np.repeat(g_long, self.num_fields))
        np.add.at(self.W[:, 1], X_flat, -self.lr * self.alpha * np.repeat(g_click, self.num_fields))
        
        # Update V
        g_interact = g_long + self.alpha * g_click
        grad_V_X = g_interact[:, None, None] * (V_sum[:, None, :] - V_X)
        np.add.at(self.V, X_flat, -self.lr * grad_V_X.reshape(-1, self.k))
        
        return np.mean(g_long**2)

    def predict(self, X):
        V_X = self.V[X]
        V_sum = np.sum(V_X, axis=1)
        interact = 0.5 * np.sum(V_sum**2 - np.sum(V_X**2, axis=1), axis=1)
        z = self.b[0] + np.sum(self.W[X, :, 0], axis=1) + interact
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="valid", choices=["train", "valid"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    splits = load_data(a.data_dir)
    enc, dim = encode_data(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    yva_long = yva[:, 0]

    model = MTL_FM(dim, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    best_score, best_state, patience = -1.0, None, 4
    bad = 0

    for ep in range(1, 16):
        idx = rng.permutation(len(ytr))
        losses = []
        for i in range(0, len(idx), 4096):
            sel = idx[i:i+4096]
            losses.append(model.step(Xtr[sel], ytr[sel]))
        
        va = evaluate(uva, yva_long, model.predict(Xva))
        print(f"Ep {ep} | Loss {np.mean(losses):.4f} | GAUC {va['GAUC']:.4f} nDCG {va['nDCG@5']:.4f} Primary {va['primary']:.4f}")
        
        if va["primary"] > best_score + 1e-5:
            best_score = va["primary"]
            best_state = (model.V.copy(), model.W.copy(), model.b.copy())
            bad = 0
        else:
            bad += 1
            if bad >= patience: break
        model.lr *= 0.9

    model.V, model.W, model.b = best_state
    X_out, _, _ = enc[a.split]
    scores = model.predict(X_out)
    
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (x, s) in enumerate(zip(splits[a.split], scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])
    print(f"Done in {time.time()-t0:.1f}s. Best Score: {best_score:.4f}")

if __name__ == "__main__":
    main()