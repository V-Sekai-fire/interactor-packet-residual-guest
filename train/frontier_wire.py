# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Frontier learning on the coded wire. Level = test clip x joint band (worst 10 / middle 10 / best 10 joints by baseline cost). Problem = one joint's 100-byte packet at one tick.
Success = the packet's ideal cost is at most the baseline median (a threshold fixed from the baseline, never from the candidate). A change stays only if total success rises and no level
loses more than 0.5 points. Tables: fitted on the 26 library avatars, scored on the 8 test avatars; ideal code length."""
import os, pickle
import numpy as np

T = os.environ["TMPDIR"].rstrip("/")
Wt = np.frombuffer(b"".join(pickle.load(open(T + "/coded_wire_train.pkl", "rb"))), np.uint8).reshape(-1, 26, 30, 100)[1:].astype(np.int64)  # our own caches
Wv = np.frombuffer(b"".join(pickle.load(open(T + "/coded_wire.pkl", "rb"))), np.uint8).reshape(-1, 8, 30, 100)[1:].astype(np.int64)
CLIPS = ["walk_1", "jog_1", "run_1", "walk_2", "circles_1", "snake_1", "walk_turns_1", "crouching_1"]
pos = np.arange(100)


def prev_zero(W): z = np.zeros_like(W); z[..., 1:] = (W[..., :-1] == 0); return z


CANDS = {
    "baseline: joint x position": lambda W: np.arange(30)[None, None, :, None] * 100 + pos,
    "A: + previous byte zero": lambda W: (np.arange(30)[None, None, :, None] * 100 + pos) * 2 + prev_zero(W),
    "B: + previous byte zero, position bands instead of joints": lambda W: ((np.arange(30)[None, None, :, None] // 5) * 100 + pos) * 2 + prev_zero(W),
}


def packet_bytes(ctxf):
    ct, cv = np.broadcast_to(ctxf(Wt), Wt.shape), np.broadcast_to(ctxf(Wv), Wv.shape); n = int(max(ct.max(), cv.max())) + 1
    cnt = np.full((n, 256), 0.05); np.add.at(cnt, (ct.ravel(), Wt.ravel()), 1)
    lp = np.log2(cnt / cnt.sum(1, keepdims=True))
    return -lp[cv, Wv].sum(-1) / 8  # (ticks, 8, 30) bytes per joint-packet


base = packet_bytes(CANDS["baseline: joint x position"])
band = np.zeros(30, int); order = np.argsort(-base.mean((0, 1))); band[order[10:20]] = 1; band[order[20:]] = 2
thr = np.median(base)
print(f"threshold (baseline median packet cost) {thr:.2f} bytes; baseline {base.sum((1, 2)).mean() / 8 * 4:.0f} B per tick of four avatars")


def success(pb):
    ok = pb <= thr
    return np.array([[ok[:, a][:, band == b].mean() * 100 for b in range(3)] for a in range(8)])  # (clip, band)


s0 = success(base)
print(f"baseline frontier success {s0.mean():.2f}%; weakest levels:", ", ".join(f"{CLIPS[a]}/band{b} {s0[a, b]:.1f}%" for a, b in sorted(np.ndindex(8, 3), key=lambda i: s0[i])[:3]))
for name, f in CANDS.items():
    if name.startswith("baseline"): continue
    pb = packet_bytes(f); s = success(pb); d = s - s0
    keep = s.mean() > s0.mean() and d.min() >= -0.5
    print(f"{name}\n   bytes/tick (4 avatars) {pb.sum((1, 2)).mean() / 8 * 4:.0f} | success {s.mean():.2f}% ({s.mean() - s0.mean():+.2f}) | worst level change {d.min():+.2f} at {CLIPS[np.unravel_index(d.argmin(), d.shape)[0]]}/band{np.unravel_index(d.argmin(), d.shape)[1]} | {'KEEP' if keep else 'DROP'}")
