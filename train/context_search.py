# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Which context makes the static byte tables smaller? The coded wire (100-byte packets, 4 avatars fitted, the other 4 scored, both ways, add-alpha smoothing, ideal code length,
no table cost). Prints the bytes per avatar-tick by byte position band, then each context's total. Contexts use only what the decoder has already read: the joint, the byte position, and
earlier bytes of the same packet or the same byte on the previous tick."""
import os, pickle
import numpy as np

T = os.environ["TMPDIR"].rstrip("/")
wire = pickle.load(open(T + "/coded_wire.pkl", "rb"))  # the coded wire written by dict_wire.py in our own TMPDIR
W = np.frombuffer(b"".join(wire), np.uint8).reshape(len(wire), 8, 30, 100)[1:]  # tick 0 is verbatim
Wi = W.astype(np.int64)
prev_tick = np.zeros_like(Wi); prev_tick[1:] = Wi[:-1]
prev_byte = np.zeros_like(Wi); prev_byte[..., 1:] = Wi[..., :-1]
prev2 = np.zeros_like(Wi); prev2[..., 2:] = Wi[..., :-2]
pos = np.broadcast_to(np.arange(100), Wi.shape).astype(np.int64)
jt = np.broadcast_to(np.arange(30)[None, None, :, None], Wi.shape).astype(np.int64)
nz_so_far = np.cumsum(Wi != 0, -1) - (Wi != 0)          # non-zero bytes before this one in the packet
q = lambda a, edges: np.digitize(a, edges)
CTX = {
    "joint x position": lambda: jt * 100 + pos,
    "+ previous byte is zero": lambda: (jt * 100 + pos) * 2 + (prev_byte == 0),
    "+ previous byte (4 classes)": lambda: (jt * 100 + pos) * 4 + q(prev_byte, [1, 16, 128]),
    "+ non-zero bytes so far (4 classes)": lambda: (jt * 100 + pos) * 4 + q(nz_so_far, [1, 3, 8]),
    "position x previous byte (no joint)": lambda: pos * 256 + prev_byte,
    "joint x position x previous 2 bytes zero": lambda: (jt * 100 + pos) * 4 + (prev_byte == 0) * 2 + (prev2 == 0),
    "+ same byte last tick zero": lambda: (jt * 100 + pos) * 2 + (prev_tick == 0),
}


def fit_score(c, tr, te, alpha=0.05):
    n = int(c.max()) + 1; cnt = np.full((n, 256), alpha)
    np.add.at(cnt, (c[:, tr].ravel(), Wi[:, tr].ravel()), 1)
    p = cnt / cnt.sum(1, keepdims=True)
    return -np.log2(p[c[:, te], Wi[:, te]])  # bits per byte, shape (ticks, 4, 30, 100)


A, B = [0, 1, 2, 3], [4, 5, 6, 7]
base = None
for name, f in CTX.items():
    c = f()
    bits = [fit_score(c, A, B), fit_score(c, B, A)]
    per = (bits[0].sum() + bits[1].sum()) / 2 / len(Wi) / 4 / 8  # bytes per avatar-tick
    if base is None:
        base = per
        bp = (bits[0].sum((0, 1, 2)) + bits[1].sum((0, 1, 2))) / 2 / len(Wi) / 4 / 8  # per byte position, bytes per avatar-tick
        print("bytes per avatar-tick by byte position (joint x position tables):")
        for lo, hi in ((0, 16), (16, 51), (51, 52), (52, 90), (90, 100)):
            print(f"  bytes {lo:2d}..{hi - 1:2d}: {bp[lo:hi].sum():6.1f}")
        print()
    print(f"{name:42s} {per:7.1f} B per avatar-tick  ({(per - base) / base * 100:+5.1f}%)")
