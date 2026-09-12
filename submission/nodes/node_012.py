import argparse, csv, os, time, collections
import numpy as np
from evaluate import evaluate

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421), 'valid': (20220422, 20220428), 'test': (20220429, 20220508)}
FIELDS = ['user_id', 'video_id', 'author_id', 'tag', 'tab', 'dur_bucket', 'like_bucket', 'finish_bucket']

def load_data(data_dir):
    vid_info = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as f:
        for r in csv.DictReader(f):
            vid_info[r['video_id']] = (r['author_id'], r['tag'])
    vid_stats = {}
    with open(os.path.join(data_dir, 'video_features_statistic_pure.csv')) as f:
        for r in csv.DictReader(f):
            s = float(r.get('show_cnt', 0) or 1)
            vid_stats[r['video_id']] = (float(r.get('like_cnt', 0) or 0)/s, float(r.get('complete_play_cnt', 0) or 0)/s)
    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vid = r['video_id']
                auth, tag = vid_info.get(vid, ('UNK', 'UNK'))
                lr, fr = vid_stats.get(vid, (0.0, 0.0))
                rows.append({'date': int(r['date']), 'user_id': r['user_id'], 'video_id': vid,
                             'author_id': auth, 'tag': tag, 'tab': r['tab'], 'duration': float(r['duration_ms']),
                             'like_rate': lr, 'finish_rate': fr, 'label': 1 if r[LABEL] != '0' else 0})
    return rows

def encode(rows):
    tr = [r for r in rows if SPLITS['train'][0] <= r['date'] <= SPLITS['train'][1]]
    def get_edges(vals): return np.quantile(vals, np.linspace(0, 1, 11)[1:-1])
    edges = {k: get_edges([r[v] for r in tr]) for k, v in [('dur', 'duration'), ('like', 'like_rate'), ('finish', 'finish_rate')]}
    vocabs = [collections.defaultdict(lambda: len(vocabs[i])) for i in range(len(FIELDS))]
    def get_feats(r):
        return [r['user_id'], r['video_id'], r['author_id'], r['tag'], r['tab'],
                str(np.searchsorted(edges['dur'], r['duration'])),
                str(np.searchsorted(edges['like'], r['like_rate'])),
                str(np.searchsorted(edges['finish'], r['finish_rate']))]
    for r in tr:
        for i, v in enumerate(get_feats(r)): _ = vocabs[i][v]
    final_vocabs = [dict(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in final_vocabs]
    offsets = np.cumsum([0] + field_dims[:-1])
    data = {}
    for name, (lo, hi) in SPLITS.items():
        rws = [r for r in rows if lo <= r['date'] <= hi]
        if not rws: continue
        X, y, uids, vids = np.zeros((len(rws), len(FIELDS)), dtype=np.int32), np.zeros(len(rws), dtype=np.float32), [], []
        for i, r in enumerate(rws):
            for j, v in enumerate(get_feats(r)):
                X[i, j] = final_vocabs[j].get(v, len(final_vocabs[j])) + offsets[j]
            y[i], _ = r['label'], uids.append(r['user_id']), vids.append(r['video_id'])
        data[name] = (X, y, uids, vids)
    return data, int(sum(field_dims))

class FM:
    def __init__(self, dim, k=16, lr=0.005, seed=0):
        rng = np.random.default_rng(seed)
        self.k, self.lr, self.b = k, lr, 0.0
        self.w = rng.normal(0, 0.01, (dim,))
        self.v = rng.normal(0, 0.01, (dim, k))
    def predict(self, X):
        lin = np.sum(self.w[X], axis=1) + self.b
        xv = self.v[X]
        sum_v = np.sum(xv, axis=1)
        inter = 0.5 * np.sum(sum_v**2 - np.sum(xv**2, axis=1), axis=1)
        return 1 / (1 + np.exp(-np.clip(lin + inter, -20, 20)))
    def step(self, X, y):
        xv = self.v[X]
        sum_v = np.sum(xv, axis=1)
        lin = np.sum(self.w[X], axis=1) + self.b
        inter = 0.5 * np.sum(sum_v**2 - np.sum(xv**2, axis=1), axis=1)
        p = 1 / (1 + np.exp(-np.clip(lin + inter, -20, 20)))
        err = (p - y)[:, None]
        dw, dv = np.zeros_like(self.w), np.zeros_like(self.v)
        np.add.at(dw, X, err)
        for i in range(X.shape[1]):
            np.add.at(dv, X[:, i], err * (sum_v - xv[:, i, :]))
        self.b -= self.lr * np.mean(err)
        self.w -= self.lr * dw / len(y)
        self.v -= self.lr * dv / len(y)
        return np.mean(-y * np.log(p + 1e-9) - (1 - y) * np.log(1 - p + 1e-9))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True); parser.add_argument("--split", default="valid")
    parser.add_argument("--out", required=True); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = load_data(args.data_dir)
    data, dim = encode(rows)
    Xtr, ytr, utr, _ = data['train']
    Xva, yva, uva, _ = data['valid']
    m = FM(dim, k=16, lr=0.008, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    best_p, best_s, patience, bad, bs = -1.0, None, 4, 0, 8192
    for ep in range(1, 31):
        idx = rng.permutation(len(ytr))
        losses = [m.step(Xtr[sel], ytr[sel]) for i in range(0, len(idx), bs) for sel in [idx[i:i+bs]]]
        res = evaluate(uva, yva, m.predict(Xva))
        print(f"Ep {ep} | Loss {np.mean(losses):.4f} | GAUC {res['GAUC']:.4f} | nDCG {res['nDCG@5']:.4f} | P {res['primary']:.4f}")
        if res['primary'] > best_p + 1e-5:
            best_p, best_s, bad = res['primary'], (m.w.copy(), m.v.copy(), m.b), 0
        elif (bad := bad + 1) >= patience: break
    if best_s: m.w, m.v, m.b = best_s
    Xt, _, ut, vt = data[args.split]
    scores = m.predict(Xt)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (uid, vid, s) in enumerate(zip(ut, vt, scores)): w.writerow([i, uid, vid, f"{s:.6f}"])

if __name__ == "__main__":
    main()