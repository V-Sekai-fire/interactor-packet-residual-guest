# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""The real rANS coder with tables fitted on more avatars. Tables from the 26 library avatars (13 clips x 2 start offsets, none a test clip), coded wire cached by more_training.py;
scored on the 8 held-out test avatars in two groups of four (real stream bytes per tick, tick 0 skipped), each tick decoded back to the exact bytes. Compare rans_wire.py: tables fitted on four test-type avatars, 420.5."""
import os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rans_wire as R

wt = pickle.load(open(R.T + "/coded_wire_train.pkl", "rb"))  # our own caches
wv = pickle.load(open(R.T + "/coded_wire.pkl", "rb"))
Wt = np.frombuffer(b"".join(wt), np.uint8).reshape(len(wt), 26, 30, 100)
Wv = np.frombuffer(b"".join(wv), np.uint8).reshape(len(wv), 8, 30, 100)
f, start = R.build_tables(Wt, list(range(26)))
ctxs = R.ctx_list(4); sizes = []; bad = 0
for grp in ([0, 1, 2, 3], [4, 5, 6, 7]):
    for t in range(1, 400):
        syms = Wv[t, grp].reshape(-1).tolist()
        buf = R.encode(syms, ctxs, f, start); bad += R.decode(buf, ctxs, f, start) != bytes(syms); sizes.append(len(buf))
print(f"tables from 26 library avatars: {np.mean(sizes):.1f} B per tick of four test avatars, {len(sizes)} ticks, {bad} decode failures (earlier tables from 4 avatars: 420.5)")
