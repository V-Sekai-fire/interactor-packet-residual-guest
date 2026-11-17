# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Learning curve for the Gated DeltaNet-2 student (state + small head, dead-reckoning skip, L1): wire bytes on held-out avatars as training
clips are added, the same number of optimiser steps at every size, random clip subsets per size. Then bytes(N) = a + b N^-alpha is fitted.
Whole clips are held out, so this measures generalisation to new motion, not memorisation. (The flat MLP is blocklisted; it is not used here.)"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_gdn as g
import gdn3

torch.set_num_threads(8)
T = os.environ["TMPDIR"].rstrip("/")

if __name__ == "__main__":
    testc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 90).astype(np.int32)
    seqs = g.tick_seqs("train")
    nclips = len(seqs) // 3
    dr = g.wire_cost([[g.stream(g.wrap16(testc[a, t + 1] - testc[a, t] - (testc[a, t] - testc[a, t - 1]))) for a in range(8)] for t in range(100, 398)])
    print(f"{nclips} training clips; dead reckoning on ticks 101..398: {dr:.0f} B/tick", flush=True)
    rows, rng = [], np.random.default_rng(3)
    STEPS = 24 * 175  # optimiser steps at full size (2765 windows / 64 * 24 epochs, rounded up), the same at every size
    for c in (2, 5, nclips):
        for rep in range(2 if c < nclips else 1):
            pick = sorted(rng.choice(nclips, size=c, replace=False)) if c < nclips else list(range(nclips))
            X, Y = gdn3.windows([seqs[3 * i + o] for i in pick for o in range(3)], 16)
            epochs = max(6, int(round(STEPS * 64 / len(X))))
            torch.manual_seed(11 + rep)
            m = gdn3.fit(gdn3.GDNS(), X, Y, epochs, "l1")
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                b = gdn3.measure("x", g.IntGDN(m), testc)
            rows.append((c, len(X), b))
            print(f"clips {c:2d} windows {len(X):5d} epochs {epochs:4d} -> {b:6.0f} B/tick ({(b/dr-1)*100:+5.1f}% against dead reckoning)", flush=True)
    np.save(T + "/learning_curve_gdn.npy", np.array(rows))