# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Is each test avatar-tick a frame of its source clip? For each of the 8 held-out avatars, the nearest frame of its clip by code distance over all 90 codes (30 joints x 3),
the frame step between ticks, and the share of ticks whose 90 codes all lie within 2 codes of a clip frame."""
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("~/src/interactor-packet-residual-guest/train"))
import rot_frontier as r
T = r.T
tc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64)
for i, name in enumerate(r.CLIPS):
    C = np.load(f"{T}/more/{name}.npy").astype(np.int64).reshape(-1, 30, 3); C[:, :, :2] *= -1
    # nearest clip frame for each avatar tick, by the code distance over all 90 values (hips included)
    d = np.abs(r.wrap16(tc[i][:, None] - C[None]))  # (400, F, 30, 3)
    dist = d.reshape(400, len(C), 90).sum(-1)
    near = dist.argmin(1); dmin = dist.min(1)
    steps = np.diff(near)
    ex = (d[np.arange(400), near].reshape(400, 90) == 0).mean(1)
    print(f"{name:13s} clip frames {len(C):5d} | matched frame steps (mode, share): {np.bincount(np.clip(steps, 0, 9)).argmax()}, {np.bincount(np.clip(steps, 0, 9)).max()/len(steps):.2f} | first/last matched frame {near[0]}, {near[-1]} | per-tick code error mean {dmin.mean()/90:6.1f}, share of ticks with every code within 2 of a clip frame {np.mean(d[np.arange(400), near].reshape(400,90).max(1) <= 2):.2f}")
