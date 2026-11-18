# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Static tables against tables that adapt as the stream is decoded. Coded wire, one avatar at a time (30 packets of 100 bytes per tick), context = joint x byte position.
Prior: tables fitted on four other avatars (add-alpha), worth w pseudo-samples per context. Adaptive: after each tick the decoded bytes are added to the counts, so the next tick's
probability is (w * prior + counts) / (w + n). Both ends can do that, since the decoder holds the same bytes. Ideal code length, bytes per avatar-tick, scored on the avatars not
in the prior, both ways. Also: no prior (adaptive from empty) and a control with the decoded bytes shuffled across ticks before adding (which must lose the adaptation)."""
import os, pickle
import numpy as np

T = os.environ["TMPDIR"].rstrip("/")
wire = pickle.load(open(T + "/coded_wire.pkl", "rb"))  # our own cache, written by dict_wire.py
W = np.frombuffer(b"".join(wire), np.uint8).reshape(len(wire), 8, 30, 100)[1:].astype(np.int64)
NT = len(W)
ctx = (np.arange(30)[:, None] * 100 + np.arange(100)[None]).reshape(-1)  # (3000,)


def prior_of(avs, alpha=0.05):
    cnt = np.full((3000, 256), alpha)
    for a in avs:
        for t in range(NT): np.add.at(cnt, (ctx, W[t, a].reshape(-1)), 1)
    return cnt / cnt.sum(1, keepdims=True)


def cost(a, prior, w, shuffle_rng=None):
    """bytes for avatar a over all ticks; prior None means adaptive from empty (add-alpha 0.05)"""
    cnt = np.zeros((3000, 256)); tot = np.zeros(3000); bits = 0.0
    order = np.arange(NT) if shuffle_rng is None else shuffle_rng.permutation(NT)
    for t in range(NT):
        sym = W[t, a].reshape(-1)
        if prior is None: p = (cnt[ctx, sym] + 0.05) / (tot + 0.05 * 256)
        else: p = (w * prior[ctx, sym] + cnt[ctx, sym]) / (w + tot)
        bits += -np.log2(p).sum()
        add = W[order[t], a].reshape(-1) if shuffle_rng is not None else sym
        cnt[ctx, add] += 1; tot += 1
    return bits / 8 / NT


A, B = [0, 1, 2, 3], [4, 5, 6, 7]
pa, pb = prior_of(A), prior_of(B)
rng = np.random.default_rng(0)
print("ideal bytes per avatar-tick (mean over the 8 avatars, each scored with a prior from the other four):")
rows = [("static prior (as the coder now)", lambda a, pr: cost(a, pr, 1e12))]
for w in (4, 16, 64, 256):
    rows.append((f"adaptive from the prior, w = {w}", lambda a, pr, w=w: cost(a, pr, w)))
rows.append(("adaptive from empty", lambda a, pr: cost(a, None, 0)))
rows.append(("control: prior w = 16, adds bytes shuffled across ticks", lambda a, pr: cost(a, pr, 16, rng)))
for name, f in rows:
    v = [f(a, pb) for a in A] + [f(a, pa) for a in B]
    print(f"  {name:56s} {np.mean(v):6.1f}   (x4 = {np.mean(v) * 4:5.0f} B per tick of four avatars)")
