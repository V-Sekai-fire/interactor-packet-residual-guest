# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Residual size class coded with a context, from the data. A residual |r| is a size class (0 for zero, else bit length) plus its low bits and a sign.
Cost = -log2 P(class | context) + (bits - 1) + 1 sign bit. Counts are fitted on four avatars and scored on the other four (and the reverse), with add-one smoothing.
Contexts: none; the same joint-component's previous size class; that plus the count of large residuals among the other joints on the previous tick. Proxy bits."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rot_frontier as r

tc = np.load(r.T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64)
c2, c1, c0 = tc[:, 99:397], tc[:, 100:398], tc[:, 101:399]
a = np.abs(r.wrap16(c0 - r.p_codes(c2, c1)))
cls = np.where(a == 0, 0, np.floor(np.log2(np.maximum(a, 1))).astype(int) + 1)  # 0..16
raw = np.where(a == 0, 0, cls)  # low bits (cls-1) plus the sign
NC = 17


def contexts(kind):
    prev = np.zeros_like(cls); prev[:, 1:] = cls[:, :-1]
    if kind == "none":
        return np.zeros_like(cls)
    if kind == "prev class":
        return prev
    jc = (np.arange(30)[None, None, :, None] * 3 + np.arange(3)[None, None, None, :]) + np.zeros_like(cls)  # joint x component
    if kind == "joint x component":
        return jc
    if kind == "joint x component + prev class":
        return jc * NC + prev
    big = (cls > 7).astype(int)  # |r| > 127
    pb = np.zeros_like(big); pb[:, 1:] = big[:, :-1]
    others = pb.sum((2, 3), keepdims=True) - pb.sum(3, keepdims=True)  # large residuals in the other joints, previous tick (all components)
    bucket = np.minimum(others // 10, 8)
    return prev * 9 + bucket


def score(kind, train_idx, test_idx):
    ctx = contexts(kind); nctx = ctx.max() + 1
    cnt = np.ones((nctx, NC))
    np.add.at(cnt, (ctx[train_idx].ravel(), cls[train_idx].ravel()), 1)
    p = cnt / cnt.sum(1, keepdims=True)
    t = -np.log2(p[ctx[test_idx], cls[test_idx]]) + raw[test_idx]
    return t.sum() / (len(test_idx) * a.shape[1])


A, B = np.arange(0, 8, 2), np.arange(1, 8, 2)
cur = r.bits_of(a).sum() / (8 * a.shape[1])
print(f"current proxy (<=126: log2|r|+2; escape 16): {cur:.0f} bits per avatar-tick")
for kind in ("none", "joint x component", "prev class", "joint x component + prev class", "prev class + others large"):
    s = (score(kind, A, B) * len(B) + score(kind, B, A) * len(A)) / 8
    print(f"class coded with context '{kind}': {s:.0f} bits per avatar-tick  ({(s - cur) / cur * 100:+.1f}%)")
# control: each joint-component's residual classes shuffled in time (the low bits follow the class); a temporal context must lose its gain, a per-joint one must keep it
rng = np.random.default_rng(0); sh = cls.copy()
for i in range(8):
    for j in range(30):
        for k in range(3):
            sh[i, :, j, k] = rng.permutation(sh[i, :, j, k])
cls = sh; raw = np.where(cls == 0, 0, cls)
for kind in ("none", "joint x component", "prev class", "joint x component + prev class"):
    s_ = (score(kind, A, B) * 4 + score(kind, B, A) * 4) / 8
    print(f"control (shuffled in time), context '{kind}': {s_:.0f} bits per avatar-tick")
