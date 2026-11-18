# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""The contract's Makima tangent (predict_kernel.tang: weights 2|p3-p2| + |p3+p2| and 2|p1-p0| + |p1+p0|, ghost slopes mirrored as 2a - b) applied to rotations for display upscaling.
Rotations are taken in the log map at the interval's first packet P1; Makima tangents and a cubic Hermite run per component. Packets are 20 Hz (the held-out clips at 60 Hz give truth at 1/3 and 2/3 of the interval).
Wait = how long after P1's time the display must wait for packets: slerp needs P2 (50 ms); the Makima ghost variant uses P-1..P2 (50 ms); the Makima variant with a real next slope uses P3 (100 ms);
Catmull-Rom uses P0..P3 (100 ms); extrapolation uses only the past (0 ms). Mean geodesic degrees over all joints, held-out-from-test library clips."""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rot_upscale as U
import rot_frontier as r

T, K = U.T, 3
EXP, LOG, MT, geo = U.EXP, U.LOG, U.MT, U.geo_deg


def tang(p0, p1, p2, p3):  # the contract's Makima tangent, on floats
    w1 = 2 * np.abs(p3 - p2) + np.abs(p3 + p2); w2 = 2 * np.abs(p1 - p0) + np.abs(p1 + p0); s = w1 + w2
    return np.where(s == 0, (p1 + p2) / 2, (w1 * p1 + w2 * p2) / np.where(s == 0, 1, s))


def hermite(s1, t1, s2, t2, f):  # position s1 = 0 at the start; tangents in per-packet-interval units
    return (f ** 3 - 2 * f ** 2 + f) * t1 + (-2 * f ** 3 + 3 * f ** 2) * s2 + (f ** 3 - f ** 2) * t2


err = {}
test = set(r.CLIPS)
for fn in sorted(glob.glob(T + "/more/*.npy")):
    if os.path.basename(fn)[:-4] in test: continue
    C = np.load(fn).astype(np.int64).reshape(-1, 30, 3); C[:, :, :2] *= -1
    if len(C) < 200: continue
    R = r.mats(r.codes(r.mats(C))); n = (len(R) - 1) // K * K
    j = np.arange(3, (n // K) - 2) * K
    P = {k: R[j + k * K] for k in range(-3, 4)}
    base = P[0]
    X = {k: LOG(MT(base) @ P[k]) for k in range(-3, 4)}  # log map at P1 (index 0 here is the interval start; X[0] = 0)
    m = {k: X[k + 1] - X[k] for k in range(-3, 3)}  # slope from packet k to k+1
    for ph in range(1, K):
        f = ph / K; truth = R[j + ph]
        def add(name, Rhat): err.setdefault(name, {}).setdefault(ph, []).append(geo(Rhat, truth).ravel())
        add("hold the last packet (0 ms)", base)
        t_ex = tang(m[-3], m[-2], m[-1], 2 * m[-1] - m[-2])  # tangent at the latest packet from the past only, mirrored ghost
        add("Makima extrapolation, gain 1 (0 ms)", base @ EXP(f * t_ex))
        add("Makima extrapolation, gain 0.875 (0 ms)", base @ EXP(0.875 * f * t_ex))
        add("extrapolate gain 0.875, last step (0 ms)", base @ EXP(0.875 * f * m[-1]))
        add("slerp P1-P2 (50 ms)", U.slerp(base, P[1], f))
        g0 = m[0]; g1 = 2 * m[0] - m[-1]  # slopes: m[-1] into P1, m[0] out of P1 (to P2), mirrored ghost beyond P2
        t1 = tang(m[-2], m[-1], m[0], g1); t2 = tang(m[-1], m[0], g1, 2 * g1 - m[0])
        add("Makima Hermite, ghost slopes (50 ms)", base @ EXP(hermite(0, t1, X[1], t2, f)))
        t1b = tang(m[-2], m[-1], m[0], m[1]); t2b = tang(m[-1], m[0], m[1], 2 * m[1] - m[0])
        add("Makima Hermite, real next slope (100 ms)", base @ EXP(hermite(0, t1b, X[1], t2b, f)))
        add("Catmull-Rom (100 ms)", U.catmull(P[-1], base, P[1], P[2], f))
print(f"{'method':44s} {'1/3 of the interval':>20s} {'2/3':>8s}   (mean degrees, all joints; p99 in brackets)")
for name, d in err.items():
    a, b = np.concatenate(d[1]), np.concatenate(d[2])
    print(f"{name:44s} {a.mean():8.2f} ({np.percentile(a, 99):5.2f}) {b.mean():8.2f} ({np.percentile(b, 99):5.2f})")
