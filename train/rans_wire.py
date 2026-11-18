# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""A static rANS coder over the coded wire, one context per (joint, byte position): the pinned-table coder. Reference implementation in Python.

Tables are integer frequencies summing to 2**PROB_BITS with every byte value at least 1, so any byte string decodes whatever the training data was (a byte the
tables never saw costs PROB_BITS bits, it is not an error). Fitted on four avatars, scored on the other four, both ways. Each tick of 4 avatars (120 packets, 12,000 bytes)
is one rANS stream. Prints real stream bytes per tick, not ideal code length, and decodes every tick back to the exact bytes.
"""
import os
import pickle
import sys

import numpy as np

PROB_BITS = 14
M = 1 << PROB_BITS
RANS_L = 1 << 23
T = os.environ["TMPDIR"].rstrip("/")


def build_tables(W, avatars, alpha=0.05):
    """W: (ticks, avatars, 30, 100) uint8. Returns freq (3000, 256) int and cumulative start (3000, 257)."""
    cnt = np.full((30, 100, 256), alpha)
    for a in avatars:
        for j in range(30):
            for b in range(100):
                cnt[j, b] += np.bincount(W[:, a, j, b], minlength=256)
    p = cnt / cnt.sum(-1, keepdims=True)
    f = 1 + np.floor(p * (M - 256)).astype(np.int64)
    f[np.arange(30)[:, None], np.arange(100)[None, :], p.argmax(-1)] += M - f.sum(-1)  # the remainder goes to the likeliest byte
    assert (f >= 1).all() and (f.sum(-1) == M).all()
    f = f.reshape(3000, 256)
    start = np.concatenate([np.zeros((3000, 1), np.int64), np.cumsum(f, 1)], 1)
    return f, start


def encode(syms, ctxs, f, start):
    """rANS, symbols in reverse so the decoder reads forward. syms, ctxs: lists of ints."""
    x = RANS_L
    out = bytearray()
    for i in range(len(syms) - 1, -1, -1):
        c, s = ctxs[i], syms[i]
        fr, st = int(f[c][s]), int(start[c][s])
        xmax = ((RANS_L >> PROB_BITS) << 8) * fr
        while x >= xmax:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // fr) << PROB_BITS) + (x % fr) + st
    for _ in range(4):
        out.append(x & 0xFF)
        x >>= 8
    return bytes(out[::-1])


def decode(buf, ctxs, f, start):
    x = int.from_bytes(buf[:4], "big")
    pos = 4
    out = []
    for c in ctxs:
        slot = x & (M - 1)
        s = int(np.searchsorted(start[c], slot, side="right")) - 1
        out.append(s)
        x = int(f[c][s]) * (x >> PROB_BITS) + slot - int(start[c][s])
        while x < RANS_L:
            x = (x << 8) | buf[pos]
            pos += 1
    if pos != len(buf):
        raise ValueError("stream length")
    return bytes(out)


def ctx_list(avatars_n):
    return [(j * 100) + b for _ in range(avatars_n) for j in range(30) for b in range(100)]


def main():
    # the cache is the coded wire dict_wire.py wrote to our own TMPDIR; it is never loaded from anywhere else
    wire = pickle.load(open(T + "/coded_wire.pkl", "rb"))
    W = np.frombuffer(b"".join(wire), np.uint8).reshape(len(wire), 8, 30, 100)
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    A, B = [0, 1, 2, 3], [4, 5, 6, 7]
    ctxs = ctx_list(4)
    tot = {}
    for name, tr, te in (("train 0-3, score 4-7", A, B), ("train 4-7, score 0-3", B, A)):
        f, start = build_tables(W, tr)
        sizes, bad = [], 0
        for t in range(1, ticks):  # tick 0 is every packet verbatim; the codec's own figures skip it too
            syms = W[t, te].reshape(-1).tolist()
            buf = encode(syms, ctxs, f, start)
            bad += decode(buf, ctxs, f, start) != bytes(syms)
            sizes.append(len(buf))
        tot[name] = np.mean(sizes)
        print(f"{name}: {np.mean(sizes):.1f} B per tick (4 avatars, 120 packets), {ticks - 1} ticks, {bad} decode failures", flush=True)
    print(f"MEAN {np.mean(list(tot.values())):.1f} B per tick against the codec's prefix figure 691 and the best zstd dictionary 583")
    # controls: a flipped stream byte must be caught or decode to something else; the stream also decodes with tables trained on the wrong avatars (lossless, only bigger)
    f, start = build_tables(W, A)
    syms = W[0, B].reshape(-1).tolist()
    buf = bytearray(encode(syms, ctxs, f, start))
    buf[len(buf) // 2] ^= 0x10
    try:
        same = decode(bytes(buf), ctxs, f, start) == bytes(syms)
    except Exception:
        same = False
    print("CONTROL a flipped stream byte is not decoded to the original:", not same)
    f2, s2 = build_tables(W, [0])  # tables from one avatar only: still lossless on others
    b2 = encode(syms, ctxs, f2, s2)
    print("CONTROL tables from one avatar only: lossless", decode(b2, ctxs, f2, s2) == bytes(syms), f"| {len(b2)} B for that tick (four-avatar tables: {len(encode(syms, ctxs, f, start))} B)")


if __name__ == "__main__":
    main()
