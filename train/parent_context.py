# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Kinematic context: code a joint packet knowing what its parent did in the same tick. Joints are listed parent first (parents.txt), so a decoder has the parent's packet before the child's.
Context = joint x byte position x class, class = number of rotation fields (swing_x, swing_z, twist: bytes 13..15 == -128) that escaped: in the parent's packet this tick (P), in the joint's
own packet last tick (L), or both (P+L, 16 classes). Static tables fitted on the 26 library avatars, scored on the 8 test avatars, ticks 101 and later, ideal bytes per avatar-tick.
Controls: the parent's count taken from a random other joint (same tick), and from the same parent at a shuffled tick."""
import os, pickle
import numpy as np

T = os.environ["TMPDIR"].rstrip("/")
parents = [int(x) for x in open(T + "/mmg/data/o3de-motionmatching/walk_1/parents.txt").read().split()]
Wt = np.frombuffer(b"".join(pickle.load(open(T + "/coded_wire_train.pkl", "rb"))), np.uint8).reshape(400, 26, 30, 100)[1:].astype(np.int64)  # our own caches
Wv = np.frombuffer(b"".join(pickle.load(open(T + "/coded_wire.pkl", "rb"))), np.uint8).reshape(400, 8, 30, 100)[1:].astype(np.int64)
rng = np.random.default_rng(0)
OTHER = (np.arange(30) + rng.integers(1, 29, 30)) % 30  # one fixed random joint per joint, for the control
SHUF = {}


def esc(W): return (W[..., 13:16] == 128).sum(-1)  # (ticks, avatars, 30) rotation fields that escaped


def classes(W, kind):
    e = esc(W); par = np.array([max(p, 0) for p in parents]); has = np.array([p >= 0 for p in parents])
    P = np.where(has, e[:, :, par], 0)
    L = np.zeros_like(e); L[1:] = e[:-1]
    if kind == "P": return P
    if kind == "L": return L
    if kind == "PL": return P * 4 + L
    if kind == "control: random other joint":
        return e[:, :, OTHER]
    if kind == "control: parent at a shuffled tick": return P[SHUF.setdefault(len(P), rng.permutation(len(P)))]
    return np.zeros_like(e)


def fit_score(Wfit, Wsc, kind, skip=100):
    nc = 16; cf, cs = classes(Wfit, kind), classes(Wsc, kind)
    cnt = np.full((30, 100, nc, 256), 0.05); j = np.arange(30)[None, None, :, None]; p = np.arange(100)[None, None, None, :]
    np.add.at(cnt, (np.broadcast_to(j, Wfit.shape).ravel(), np.broadcast_to(p, Wfit.shape).ravel(), np.broadcast_to(cf[..., None], Wfit.shape).ravel(), Wfit.ravel()), 1)
    lp = np.log2(cnt / cnt.sum(-1, keepdims=True))
    W = Wsc[skip:]; cc = np.broadcast_to(cs[..., None], Wsc.shape)[skip:]
    return -lp[np.broadcast_to(j, W.shape), np.broadcast_to(p, W.shape), cc, W].sum(-1).sum(-1).mean() / 8


def score(kind): return fit_score(Wt, Wv, kind)


def cross(kind): return (fit_score(Wv[:, :4], Wv[:, 4:], kind) + fit_score(Wv[:, 4:], Wv[:, :4], kind)) / 2


for label, fn in (("tables fitted on the 26 library avatars", score), ("tables fitted on four test-type avatars, scored on the other four (both ways)", cross)):
    base = fn("none"); print(f"\n{label}\n  static tables, joint x byte position: {base:.1f} bytes per avatar-tick ({base * 4:.0f} per tick of four)")
    for k, name in (("P", "+ parent's rotation escapes this tick"), ("L", "+ own rotation escapes last tick"), ("PL", "+ both"), ("control: random other joint", "control: a fixed random other joint's escapes"), ("control: parent at a shuffled tick", "control: the parent's escapes at a shuffled tick")):
        v = fn(k); print(f"  {name:55s} {v:6.1f}  ({(v - base) / base * 100:+5.1f}%)")
