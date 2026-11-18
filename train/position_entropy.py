# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""What a static coder could reach on the real coded wire. Each byte of a coded 100-byte packet is coded with a context and a table fitted on four avatars,
scored on the other four (both ways), add-alpha smoothing. Contexts: none; byte position (0..99); joint x byte position; joint x byte position x whether the
same byte was zero on the previous tick. Cost is the ideal code length, no table cost (the tables are fixed in advance). Needs the cached wire from the codec."""
import os, pickle
import numpy as np

T = os.environ["TMPDIR"].rstrip("/")
wire = pickle.load(open(T + "/coded_wire.pkl", "rb"))
W = np.frombuffer(b"".join(wire), np.uint8).reshape(len(wire), 8, 30, 100)  # tick, avatar, joint, byte
prev = np.zeros_like(W); prev[1:] = W[:-1]
pz = (prev == 0).astype(np.int64)
pos = np.broadcast_to(np.arange(100), W.shape)
jt = np.broadcast_to(np.arange(30)[None, None, :, None], W.shape)


def ctx(kind):
    if kind == "none": return np.zeros(W.shape, np.int64)
    if kind == "byte position": return pos.astype(np.int64)
    if kind == "joint x byte position": return (jt * 100 + pos).astype(np.int64)
    if kind == "joint x byte position x prev zero": return ((jt * 100 + pos) * 2 + pz).astype(np.int64)


def score(c, tr, te, alpha=0.05):
    n = c.max() + 1; cnt = np.full((n, 256), alpha)
    np.add.at(cnt, (c[:, tr].ravel(), W[:, tr].ravel()), 1)
    p = cnt / cnt.sum(1, keepdims=True)
    return -np.log2(p[c[:, te], W[:, te]]).sum() / (len(wire) * len(te)) / 8  # bytes per avatar-tick


A, B = [0, 1, 2, 3], [4, 5, 6, 7]
print("bytes per avatar-tick (30 entities), tables fitted on four avatars and scored on the other four, mean of both ways:")
for kind in ("none", "byte position", "joint x byte position", "joint x byte position x prev zero"):
    c = ctx(kind); s = (score(c, A, B) + score(c, B, A)) / 2
    print(f"  {kind:36s} {s:7.1f} B   ({s * 4:6.0f} B per tick for 4 avatars)")
c = ctx("joint x byte position x prev zero"); sh = W.copy()
rng = np.random.default_rng(0)
for a in range(8):
    for j in range(30):
        for b in range(100): sh[:, a, j, b] = rng.permutation(sh[:, a, j, b])
W_real = W; W = sh
s = (score(c, A, B) + score(c, B, A)) / 2
print(f"  control, each byte shuffled in time, same context: {s:7.1f} B   ({s * 4:6.0f} per tick for 4 avatars)")
