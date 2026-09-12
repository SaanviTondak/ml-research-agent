import argparse
import csv
import os
import time
from collections import defaultdict
import numpy as np

from evaluate import evaluate

HEADER = ["row_id", "user_id", "video_id", "score"]
LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tag', 'video_type', 
          'upload_type', 'music_type', 'play_progress', 'user_active', 'follow_range',
          'hour', 'day_of_week', 'complete_rate', 'like_rate']

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

def load_custom(data_dir):
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
            sc = float(r.get('show_cnt', 0)) + 1.0
            vid_stats[r['video_id']] = {
                'play_progress': float(r.get('play_progress', 0.0)),
                'complete_rate': float(r.get('complete_play_cnt', 0)) / sc,
                'like_rate': float(r.get('like_cnt', 0)) / sc
            }

    user_features = {}
    with open(os.path.join(data_dir, 'user_features_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            user_features[r['user_id']] = {
                'user_active': r['user_active_degree'],
                'follow_range': r['follow_user_num_range']
            }

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vid, uid = r['video_id'], r['user_id']
                vf = vid_features.get(vid, {'author_id': 'UNK', 'tag': 'UNK', 'video_type': 'UNK', 'upload_type': 'UNK', 'music_type': 'UNK'})
                uf = user_features.get(uid, {'user_active': 'UNK', 'follow_range': 'UNK'})
                vs = vid_stats.get(vid, {'play_progress': 0.0, 'complete_rate': 0.0, 'like_rate': 0.0})
                rows.append((
                    int(r['date']), uid, vid, vf['author_id'], r['tab'], float(r['duration_ms']),
                    1 if r[LABEL] != '0' else 0, vf['tag'], vf['video_type'], vf['upload_type'],
                    vf['music_type'], vs['play_progress'], uf['user_active'], uf['follow_range'],
                    int(r['hourmin']) // 100, int(r['date']) % 7, vs['complete_rate'], vs['like_rate']
                ))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def encode_custom(splits):
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr])
    prog_edges = _bucket_edges([x[11] for x in tr])
    comp_edges = _bucket_edges([x[16] for x in tr])
    like_edges = _bucket_edges([x[17] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(dur_edges, x[5]))),
                x[7], x[8], x[9], x[10], str(int(np.searchsorted(prog_edges, x[11]))),
                x[12], x[13], str(x[14]), str(x[15]),
                str(int(np.searchsorted(comp_edges, x[16]))), str(int(np.searchsorted(like_edges, x[17])))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]: vocabs[i][v] = len(vocabs[i])
    
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
            y[n], users.append(x[6]), x[1]
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))

class ListwiseFM:
    def __init__(self, num_features, num_fields, k=20, lr=0.001, wd=1e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.k, self.lr, self.wd, self.F = k, lr, wd, num_fields
        self.V = rng.normal(0.0, 0.01, size=(num_features, k)).astype(np.float32)
        self.W = np.zeros(num_features, dtype=np.float32)
        self.b = np.float32(0.0)
        self.mV, self.vV = np.zeros_like(self.V), np.zeros_like(self.V)
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.t = 0

    def predict(self, X):
        V_X = self.V[X]
        sum_V = np.sum(V_X, axis=1)
        inter = 0.5 * np.sum(sum_V**2 - np.sum(V_X**2, axis=1), axis=1)
        return np.sum(self.W[X], axis=1) + self.b + inter

    def step_listwise(self, X_batch, batch_users):
        V_batch = self.V[X_batch]
        sum_V_batch = np.sum(V_batch, axis=1)
        scores_batch = np.sum(self.W[X_batch], axis=1) + self.b + 0.5 * np.sum(sum_V_batch**2 - np.sum(V_batch**2, axis=1), axis=1)

        grad_batch = np.zeros(len(X_batch), dtype=np.float32)
        total_loss, start = 0.0, 0
        for indices, pos_idx, n_pos in batch_users:
            u_len = len(indices)
            u_scores = scores_batch[start:start+u_len]
            exps = np.exp(u_scores - np.max(u_scores))
            sum_exps = np.sum(exps)
            total_loss -= np.sum(u_scores[pos_idx] - (np.max(u_scores) + np.log(sum_exps)))
            grad_u = n_pos * (exps / sum_exps)
            grad_u[pos_idx] -= 1.0
            grad_batch[start:start+u_len] = grad_u / len(batch_users)
            start += u_len

        dV, dW = np.zeros_like(self.V), np.zeros_like(self.W)
        np.add.at(dW, X_batch.ravel(), np.repeat(grad_batch, self.F))
        grad_V_batch = (grad_batch[:, None, None] * (sum_V_batch[:, None, :] - V_batch)).reshape(-1, self.k)
        np.add.at(dV, X_batch.ravel(), grad_V_batch)

        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - 0.999**self.t) / (1.0 - 0.9**self.t)
        if self.wd > 0: dV += self.wd * self.V; dW += self.wd * self.W
        self.mV = 0.9 * self.mV + 0.1 * dV
        self.vV = 0.999 * self.vV + 0.001 * (dV**2)
        self.V -= lr_t * self.mV / (np.sqrt(self.vV) + 1e-8)
        self.mW = 0.9 * self.mW + 0.1 * dW
        self.vW = 0.999 * self.vW + 0.001 * (dW**2)
        self.W -= lr_t * self.mW / (np.sqrt(self.vW) + 1e-8)
        return total_loss / len(batch_users)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--split", default="valid")
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=20); ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()

    splits = load_custom(a.data_dir)
    enc, dim = encode_custom(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]

    u2idx = defaultdict(list)
    for i, u in enumerate(utr): u2idx[u].append(i)
    valid_u_idx = []
    for u, idxs in u2idx.items():
        idxs = np.array(idxs, dtype=np.int32)
        pos_mask = (ytr[idxs] == 1); n_pos = np.sum(pos_mask)
        if 0 < n_pos < len(idxs): valid_u_idx.append((idxs, np.where(pos_mask)[0], n_pos))

    m = ListwiseFM(dim, len(FIELDS), k=a.k, lr=a.lr, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, a.epochs + 1):
        te = time.time(); rng.shuffle(valid_u_idx); losses = []
        for i in range(0, len(valid_u_idx), a.bs):
            bu = valid_u_idx[i:i + a.bs]
            losses.append(m.step_listwise(Xtr[np.concatenate([u[0] for u in bu])], bu))
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"ep {ep:2d} | loss {np.mean(losses):.4f} | GAUC {va['GAUC']:.4f} nDCG {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-te:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0; best_state = (m.V.copy(), m.W.copy(), m.b)
        else:
            bad += 1
            if bad >= 5: break

    m.V, m.W, m.b = best_state
    X = enc[a.split][0]
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(HEADER)
        for i, (x, s) in enumerate(zip(splits[a.split], m.predict(X))):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])

if __name__ == "__main__":
    main()