# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Several pinned table sets, one chosen per avatar and signalled (log2 K bits per avatar-segment). Leave one avatar out: the other seven are grouped into K sets by hard-assignment
clustering on code length (start from a random split, assign each avatar to the set that codes it shortest, refit, repeat), the held-out avatar is coded with the best of the K sets
(the choice costs log2 K bits, here charged once per avatar over the whole clip, the most optimistic case). Ideal code length, joint x byte position contexts, bytes per avatar-tick.
Only 8 avatars, so this is indicative. Control: the same with the sets built from a random split of the seven (no clustering)."""
import os, pickle
import numpy as np

T = os.environ["TMPDIR"].rstrip("/")
wire = pickle.load(open(T + "/coded_wire.pkl", "rb"))  # our own cache, written by dict_wire.py
W = np.frombuffer(b"".join(wire), np.uint8).reshape(len(wire), 8, 30, 100)[1:].astype(np.int64)
NT = len(W)
ctx = (np.arange(30)[:, None] * 100 + np.arange(100)[None]).reshape(-1)
cnt = np.zeros((8, 3000, 256))
for a in range(8):
    for t in range(NT): np.add.at(cnt[a], (ctx, W[t, a].reshape(-1)), 1)


def table(members, alpha=0.05):
    c = cnt[members].sum(0) + alpha
    return np.log2(c / c.sum(1, keepdims=True))


def bytes_of(a, lp):  # ideal bytes per avatar-tick of avatar a under log2-probabilities lp
    return -(cnt[a] * lp).sum() / 8 / NT


def sets_for(train, K, rng, cluster=True):
    train = list(train)
    assign = rng.permutation(np.arange(len(train)) % K)
    for _ in range(12 if cluster else 0):
        lps = [table([train[i] for i in range(len(train)) if assign[i] == k]) if (assign == k).any() else None for k in range(K)]
        new = np.array([int(np.argmin([bytes_of(train[i], lp) if lp is not None else 1e9 for lp in lps])) for i in range(len(train))])
        if (new == assign).all(): break
        assign = new
    return [table([train[i] for i in range(len(train)) if assign[i] == k]) for k in range(K) if (assign == k).any()]


rng = np.random.default_rng(3)
print("leave one avatar out, ideal bytes per avatar-tick (mean over the 8 held-out avatars):")
for K in (1, 2, 3):
    for cluster, label in ((True, "clustered"), (False, "random split")):
        if K == 1 and not cluster: continue
        vals = []
        for h in range(8):
            sets = sets_for([a for a in range(8) if a != h], K, rng, cluster)
            vals.append(min(bytes_of(h, lp) for lp in sets) + np.log2(len(sets)) / 8 / NT)  # the choice's bits, once per clip
        print(f"  K = {K} {label:13s} {np.mean(vals):6.1f}   (x4 = {np.mean(vals) * 4:5.0f} B per tick of four avatars)")
