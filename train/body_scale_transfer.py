# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Do tables fitted on library avatars carry over to avatars of other body sizes? The 8 test clips are coded again with every joint position (and so velocity and acceleration) scaled by
stature / median stature, the statures drawn from the real random-body samples of the GarmentCodeData v2 parquet (`body_measurements`, height in cm; numbers only, no body meshes). Rotations are
unchanged. Cost = ideal bytes per avatar-tick under static joint x byte-position tables fitted on the 26 unscaled library avatars (coded_wire_train.pkl), ticks 101 and later. Control: scale 1.0
reproduces the unscaled cost. Needs frames_ones.bin, frames_varied.bin (mm_to_packets with a SCALES list) in TMPDIR and PACKET_MODULE_DIR."""
import os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import packet_residual_lossless as P

T = os.environ["TMPDIR"].rstrip("/")
Wt = np.frombuffer(b"".join(pickle.load(open(T + "/coded_wire_train.pkl", "rb"))), np.uint8).reshape(400, 26, 3000)[1:].astype(np.int64)  # our own cache
cnt = np.full((3000, 256), 0.05); ctx = np.tile(np.arange(3000), (399, 1))
for a in range(26): np.add.at(cnt, (ctx.ravel(), Wt[:, a].ravel()), 1)
lp = np.log2(cnt / cnt.sum(1, keepdims=True))
scales = [float(x) for x in open(T + "/scales.txt").read().split(",")]


def per_avatar(path):
    wire, bad, verb, tot, _ = P.run(open(path, "rb").read(), 240, 400, "kernel")
    W = np.frombuffer(b"".join(wire), np.uint8).reshape(400, 8, 3000)[101:].astype(np.int64)
    return -lp[np.arange(3000)[None, None, :], W].sum(-1).mean(0) / 8, bad  # bytes per avatar-tick, per avatar


ones, b0 = per_avatar(T + "/frames_ones.bin"); var, b1 = per_avatar(T + "/frames_varied.bin")
print(f"codec decode failures: {b0} (scale 1.0), {b1} (varied)")
print("avatar  clip           scale   ideal bytes at scale 1.0   at its own scale   change")
names = ["walk_1", "jog_1", "run_1", "walk_2", "circles_1", "snake_1", "walk_turns_1", "crouching_1"]
for a in range(8): print(f"  {a}     {names[a]:13s}  {scales[a]:.3f}   {ones[a]:8.1f}                 {var[a]:8.1f}          {(var[a] - ones[a]) / ones[a] * 100:+5.1f}%")
print(f"mean over the 8 avatars: {ones.mean():.1f} at scale 1.0 ({ones.mean() * 4:.0f} per tick of four), {var.mean():.1f} resized ({var.mean() * 4:.0f} per tick of four), {(var.mean() - ones.mean()) / ones.mean() * 100:+.1f}%")
