# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Fit the static tables on N library avatars (clips that are not test clips) and score them on the 8 held-out test avatars, ideal bytes per avatar-tick.
Needs frames_train.bin from mm_to_packets.py (26 avatars, 13 clips) and PACKET_MODULE_DIR; caches the coded wire in TMPDIR. Run from tools/."""
import os, pickle, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import numpy as np
import packet_residual_lossless as P
T = os.environ["TMPDIR"].rstrip("/"); cache = T + "/coded_wire_train.pkl"
if os.path.exists(cache): wire = pickle.load(open(cache, "rb"))  # our own cache
else:
    data = open(T + "/frames_train.bin", "rb").read(); t0 = time.time()
    wire, bad, verb, tot, _ = P.run(data, 780, 400, "kernel"); print("encoded", tot, "packets, bad", bad, f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump(wire, open(cache, "wb"))
Wt = np.frombuffer(b"".join(wire), np.uint8).reshape(len(wire), 26, 30, 100)[1:].astype(np.int64)
Wv = np.frombuffer(b"".join(pickle.load(open(T + "/coded_wire.pkl", "rb"))), np.uint8).reshape(400, 8, 30, 100)[1:].astype(np.int64)
ctx = (np.arange(30)[:, None] * 100 + np.arange(100)[None]).reshape(-1)
def counts(Wx, avs):
    c = np.zeros((3000, 256))
    for a in avs:
        for t in range(len(Wx)): np.add.at(c, (ctx, Wx[t, a].reshape(-1)), 1)
    return c
cv = counts(Wv, range(8)); ct = [counts(Wt, [a]) for a in range(26)]
rng = np.random.default_rng(0)
def bytes_per(avs, alpha=0.05):
    c = sum(ct[a] for a in avs) + alpha; lp = np.log2(c / c.sum(1, keepdims=True))
    return -(cv * lp).sum() / 8 / len(Wv) / 8
print("tables fitted on N library avatars (other clips), scored on the 8 held-out test avatars, ideal bytes per avatar-tick:")
for n in (4, 8, 16, 26):
    v = [bytes_per(rng.choice(26, n, replace=False)) for _ in range(5 if n < 26 else 1)]
    print(f"  N = {n:2d}: {np.mean(v):6.1f}  (x4 = {np.mean(v) * 4:4.0f} B per tick of four avatars)")
cA = sum(counts(Wv, [a]) for a in range(4)) + 0.05; lpA = np.log2(cA / cA.sum(1, keepdims=True))
print(f"for reference, the earlier tables fitted on 4 test-type avatars and scored on the other 4: ~100.4")
both = sum(ct) + cv + 0.05
