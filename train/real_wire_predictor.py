# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""A learned rotation predictor inside the real 100-byte codec, scored by the real rANS stream. Predictions from phase_manifold.py (PAE_DUMP, shape 8 x 400 x 90, trained without the test
clips) replace the codec's p1 + (p1 - p2) for swing_x, swing_z and twist, looked up by (gid, hlc); ticks before 101 and any packet without an entry keep the codec's own rule. Everything
else in the packet is untouched. The coded wire (tick x avatar x joint x 100 bytes) goes through rans_wire.py: tables fitted on four avatars, scored on the other four, both ways,
every tick decoded and compared. Predictor settings: the codec's own; the learned predictions; control: each avatar gets the NEXT avatar's predictions (must be worse).
Run from the repo root with PACKET_MODULE_DIR set; needs $TMPDIR/frames.bin and the dump file given as argv[1]."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import packet_residual_lossless as P
import rans_wire as R

T = os.environ["TMPDIR"].rstrip("/")
data = open(T + "/frames.bin", "rb").read(); N = len(data) // 100
pred = np.load(sys.argv[1]).reshape(8, 400, 30, 3)
key = []  # packet index -> (gid, hlc, avatar, joint, tick)
for i in range(N):
    v = P.vals(data[i * 100:(i + 1) * 100]); t, rest = divmod(i, 240); a, j = divmod(rest, 30); key.append((v["gid"], v["hlc"], a, j, t))
ROTI = {"swing_x": 0, "swing_z": 1, "twist": 2}
orig = P.predict


def make(table):
    def predict(f, h, cur, mode):
        if f in ROTI and cur is not None and (cur.get("gid"), cur.get("hlc")) in table:
            return P.to_range(int(table[(cur["gid"], cur["hlc"])][ROTI[f]]), f)
        return orig(f, h, cur, mode)
    return predict


def table_for(shift):
    return {(g, hl): pred[(a + shift) % 8, t, j] for g, hl, a, j, t in key if t >= 101}


def stream(table):
    P.predict = make(table)
    wire, bad, verb, tot, _ = P.run(data, 240, 400, "kernel")
    P.predict = orig
    W = np.frombuffer(b"".join(wire), np.uint8).reshape(len(wire), 8, 30, 100)
    return W, bad, verb


def rans_cost(W):
    ctxs = R.ctx_list(4); out = []; fails = 0
    for fit, sc in (([0, 1, 2, 3], [4, 5, 6, 7]), ([4, 5, 6, 7], [0, 1, 2, 3])):
        f, start = R.build_tables(W, fit)
        for t in range(101, 400):  # the ticks where the learned predictor is active
            syms = W[t, sc].reshape(-1).tolist(); buf = R.encode(syms, ctxs, f, start)
            fails += R.decode(buf, ctxs, f, start) != bytes(syms); out.append(len(buf))
    return np.mean(out), fails


for name, table in (("the codec's own rule (p1 + (p1 - p2))", {}), ("learned predictions", table_for(0)), ("control: the next avatar's predictions", table_for(1))):
    W, bad, verb = stream(table); b, fails = rans_cost(W)
    print(f"{name:42s} {b:6.1f} B per tick of four avatars, ticks 101-399 | packets sent whole {verb} | stream decode failures {fails} | codec decode failures {bad}", flush=True)
