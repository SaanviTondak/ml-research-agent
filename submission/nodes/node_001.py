import argparse
import csv
import time
import os
import collections
import numpy as np

from evaluate import evaluate

HEADER = ["row_id", "user_id", "video_id", "score"]

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}

FIELDS = [
    'user_id', 'video_id', 'author_id', 'tab', 'dur_bucket',
    'hist_author_1', 'hist_author_2', 'hist_author_3',
    'hist_dur_1', 'hist_dur_2'
]


def write_scores(path, rows, scores):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])


def load_with_history(data_dir):
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    raw_rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                raw_rows.append((
                    int(r['date']), r['user_id'], r['video_id'],
                    vid2author.get(r['video_id'], 'UNK'), r['tab'],
                    float(r['duration_ms']), 1 if r[LABEL] != '0' else 0
                ))

    train_durs = [x[5] for x in raw_rows if SPLITS['train'][0] <= x[0] <= SPLITS['train'][1]]
    edges = np.quantile(np.asarray(train_durs), np.linspace(0, 1, 10 + 1)[1:-1])

    def get_dur_bucket(dur):
        return str(int(np.searchsorted(edges, dur)))

    user_author_hist = collections.defaultdict(list)
    user_dur_hist = collections.defaultdict(list)

    splits_rows = {k: [] for k in SPLITS}

    for x in raw_rows:
        dt, uid, vid, aid, tab, dur, label = x
        dur_b = get_dur_bucket(dur)

        a_hist = user_author_hist[uid]
        d_hist = user_dur_hist[uid]

        h_a1 = a_hist[-1] if len(a_hist) >= 1 else 'UNK'
        h_a2 = a_hist[-2] if len(a_hist) >= 2 else 'UNK'
        h_a3 = a_hist[-3] if len(a_hist) >= 3 else 'UNK'

        h_d1 = d_hist[-1] if len(d_hist) >= 1 else 'UNK'
        h_d2 = d_hist[-2] if len(d_hist) >= 2 else 'UNK'

        row_tuple = (dt, uid, vid, aid, tab, dur_b, h_a1, h_a2, h_a3, h_d1, h_d2, label)

        for sname, (lo, hi) in SPLITS.items():
            if lo <= dt <= hi:
                splits_rows[sname].append(row_tuple)
                break

        if label == 1:
            user_author_hist[uid].append(aid)
            user_dur_hist[uid].append(dur_b)

    return splits_rows


def encode_custom(splits):
    tr = splits['train']

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        raw_feat = x[1:11]
        for i, v in enumerate(raw_feat):
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
            raw_feat = x[1:11]
            for i, v in enumerate(raw_feat):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[11]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))


class MultiPairFM:
    def __init__(self, num_features, k=16, weight_decay=1e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.num_features = num_features
        self.k = k
        self.wd = weight_decay

        self.V = rng.normal(0.0, 0.01, size=(num_features, k)).astype(np.float32)
        self.W = np.zeros(num_features, dtype=np.float32)
        self.b = np.float32(0.0)

        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.float32(0.0)
        self.vb = np.float32(0.0)
        self.t = 0

    def predict(self, X):
        V_X = self.V[X]
        sum_V = np.sum(V_X, axis=1)
        sum_sq_V = np.sum(V_X ** 2, axis=1)
        inter = 0.5 * np.sum(sum_V ** 2 - sum_sq_V, axis=1)
        lin = np.sum(self.W[X], axis=1) + self.b
        return lin + inter

    def _forward(self, X):
        V_X = self.V[X]
        sum_V = np.sum(V_X, axis=1)
        sum_sq_V = np.sum(V_X ** 2, axis=1)
        inter = 0.5 * np.sum(sum_V ** 2 - sum_sq_V, axis=1)
        lin = np.sum(self.W[X], axis=1) + self.b
        return lin + inter, V_X, sum_V

    def step_multi_bpr(self, X_pos, X_neg, lr):
        B, F = X_pos.shape
        M = X_neg.shape[1]

        s_pos, V_pos, sum_V_pos = self._forward(X_pos)
        X_neg_flat = X_neg.reshape(-1, F)
        s_neg_flat, V_neg, sum_V_neg = self._forward(X_neg_flat)
        s_neg = s_neg_flat.reshape(B, M)

        diff = s_pos[:, None] - s_neg
        sig_neg_diff = 1.0 / (1.0 + np.exp(np.clip(diff, -30.0, 30.0)))
        loss = np.mean(np.log(1.0 + np.exp(np.clip(-diff, -30.0, 30.0))))

        dL_diff = -sig_neg_diff / (B * M)

        dL_spos = np.sum(dL_diff, axis=1)
        dL_sneg = dL_diff.ravel()

        dV = np.zeros_like(self.V)
        dW = np.zeros_like(self.W)
        db = np.float32(np.sum(dL_spos) + np.sum(dL_sneg))

        np.add.at(dW, X_pos.ravel(), np.repeat(dL_spos, F))
        grad_V_pos = dL_spos[:, None, None] * (sum_V_pos[:, None, :] - V_pos)
        np.add.at(dV, X_pos.ravel(), grad_V_pos.reshape(-1, self.k))

        np.add.at(dW, X_neg_flat.ravel(), np.repeat(dL_sneg, F))
        grad_V_neg = dL_sneg[:, None, None] * (sum_V_neg[:, None, :] - V_neg)
        np.add.at(dV, X_neg_flat.ravel(), grad_V_neg.reshape(-1, self.k))

        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr_t = lr * np.sqrt(1.0 - beta2 ** self.t) / (1.0 - beta1 ** self.t)

        if self.wd > 0:
            dV += self.wd * self.V
            dW += self.wd * self.W

        self.mV = beta1 * self.mV + (1.0 - beta1) * dV
        self.vV = beta2 * self.vV + (1.0 - beta2) * (dV ** 2)
        self.V -= lr_t * self.mV / (np.sqrt(self.vV) + eps)

        self.mW = beta1 * self.mW + (1.0 - beta1) * dW
        self.vW = beta2 * self.vW + (1.0 - beta2) * (dW ** 2)
        self.W -= lr_t * self.mW / (np.sqrt(self.vW) + eps)

        self.mb = beta1 * self.mb + (1.0 - beta1) * db
        self.vb = beta2 * self.vb + (1.0 - beta2) * (db ** 2)
        self.b -= lr_t * self.mb / (np.sqrt(self.vb) + eps)

        return float(loss)


def prepare_multi_neg_sampler(ytr, users, rng, M=4):
    user_neg = collections.defaultdict(list)
    for idx, (y, u) in enumerate(zip(ytr, users)):
        if y == 0:
            user_neg[u].append(idx)

    all_neg = np.where(ytr == 0)[0]
    all_pos = np.where(ytr == 1)[0]

    pos_to_neg_pools = []
    for pos_idx in all_pos:
        u = users[pos_idx]
        negs = user_neg[u]
        if len(negs) > 0:
            pos_to_neg_pools.append(np.array(negs, dtype=np.int32))
        else:
            pos_to_neg_pools.append(all_neg)

    def sample():
        n_pos = len(all_pos)
        sampled_neg = np.empty((n_pos, M), dtype=np.int32)
        for i, pool in enumerate(pos_to_neg_pools):
            if len(pool) >= M:
                sampled_neg[i] = rng.choice(pool, size=M, replace=False)
            else:
                sampled_neg[i] = rng.choice(pool, size=M, replace=True)
        perm = rng.permutation(n_pos)
        return all_pos[perm], sampled_neg[perm]

    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6)
    a = ap.parse_args()

    t0 = time.time()
    splits = load_with_history(a.data_dir)
    print(f"loaded {a.data_dir}: "
          + ", ".join(f"{k}={len(v):,d}" for k, v in splits.items())
          + f"  ({time.time()-t0:.1f}s)")

    enc, dim = encode_custom(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]

    rng = np.random.default_rng(a.seed)
    sampler = prepare_multi_neg_sampler(ytr, utr, rng, M=a.M)

    m = MultiPairFM(dim, k=a.k, weight_decay=a.wd, seed=a.seed)
    best, best_state, bad = -1.0, None, 0
    current_lr = a.lr

    for ep in range(1, a.epochs + 1):
        te = time.time()
        pos_idxs, neg_idxs = sampler()
        losses = []
        for i in range(0, len(pos_idxs), a.bs):
            b_pos = pos_idxs[i:i + a.bs]
            b_neg = neg_idxs[i:i + a.bs]
            loss = m.step_multi_bpr(Xtr[b_pos], Xtr[b_neg], current_lr)
            losses.append(loss)

        va = evaluate(uva, yva, m.predict(Xva))
        print(f"  epoch {ep:2d} | BPR loss {np.mean(losses):.4f} "
              f"| valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
              f"primary {va['primary']:.4f} | {time.time()-te:.1f}s")

        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop at epoch {ep}")
                break

        current_lr *= 0.88

    m.V, m.W, m.b = best_state
    print(f"  best valid primary {best:.4f}")

    rows = splits[a.split]
    if not rows:
        raise SystemExit(f"split '{a.split}' is empty in {a.data_dir} - nothing to score")

    X = enc[a.split][0]
    write_scores(a.out, rows, m.predict(X))
    print(f"wrote {a.out}: {len(rows):,d} rows (split={a.split}) "
          f"in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()