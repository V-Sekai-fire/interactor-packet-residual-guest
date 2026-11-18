# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Rotation upscaling: the receiver holds packets at a low rate and must show a higher one. Ground truth is the real 30 fps mocap (no real 64 Hz motion exists on
disk), so the ratio tested is 3 (keep every 3rd frame, rebuild the two between), close to 64/20 = 3.2. Error is the geodesic angle to the true frame, in degrees,
after both go through the packet's code quantisation. Interpolation needs the next packet (one packet of latency); extrapolation does not."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rot_frontier as r

T = r.T
MT, LOG, EXP = r.mT, r.log_so3, r.exp_so3


def geo_deg(A, B):
    return np.degrees(np.linalg.norm(LOG(MT(A) @ B), axis=-1))


def slerp(Ra, Rb, f):
    return Ra @ EXP(f * LOG(MT(Ra) @ Rb))


def catmull(R0, R1, R2, R3, f, tension=0.5):  # Catmull-Rom in the tangent space at R1
    p0, p1, p2, p3 = LOG(MT(R1) @ R0), np.zeros_like(LOG(MT(R1) @ R0)), LOG(MT(R1) @ R2), LOG(MT(R1) @ R3)
    t1, t2 = tension * (p2 - p0), tension * (p3 - p1)
    h00, h10, h01, h11 = 2 * f**3 - 3 * f**2 + 1, f**3 - 2 * f**2 + f, -2 * f**3 + 3 * f**2, f**3 - f**2
    return R1 @ EXP(h00 * p1 + h10 * t1 + h01 * p2 + h11 * t2)


def code_lerp(Ca, Cb, f):  # the naive route: lerp the integer codes (twist wrapped)
    d = Cb - Ca; d[..., 2] = r.wrap16(d[..., 2])
    return r.mats(np.round(Ca + f * d).astype(np.int64))


def run(clips, K=3):
    err = {}
    def add(name, A, B):
        err.setdefault(name, []).append(geo_deg(A, B))
    for C in clips:
        R = r.mats(r.codes(r.mats(C)))  # through the code quantisation, as on the wire
        n = (len(R) - 1) // K * K
        j = np.arange(1, (n // K) - 1) * K          # keyframes with a neighbour on each side
        for ph in range(1, K):
            f = ph / K
            truth = R[j + ph]
            R0, R1, R2, R3 = R[j - K], R[j], R[j + K], R[np.minimum(j + 2 * K, len(R) - 1)]
            add("hold the last packet", R1, truth)
            add("code-lerp (interpolate)", code_lerp(r.codes(R1), r.codes(R2), f), truth)
            add("slerp (interpolate)", slerp(R1, R2, f), truth)
            add("Catmull-Rom on SO(3) (interpolate)", catmull(R0, R1, R2, R3, f), truth)
            for a in (1.0, 0.875, 0.75):
                add(f"extrapolate a={a} (causal)", R1 @ EXP(a * f * LOG(MT(R0) @ R1)), truth)
    return {k: np.concatenate([x.reshape(-1, 30) for x in v]) for k, v in err.items()}


if __name__ == "__main__":
    test = set(r.CLIPS)
    clips = []
    for f in sorted(glob.glob(T + "/more/*.npy")):
        C = np.load(f).astype(np.int64).reshape(-1, 30, 3); C[:, :, :2] *= -1
        if len(C) > 200 and os.path.basename(f)[:-4] not in test: clips.append(C)
    print(len(clips), "clips, ratio 3 (30 fps frames kept every 3rd, 2 rebuilt between)")
    e = run(clips)
    tc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64)
    band = np.zeros(30, int)
    d = np.abs(r.wrap16(tc[:, 2:] - 2 * tc[:, 1:-1] + tc[:, :-2])); worst = np.argsort(-r.bits_of(d).sum((0, 1, 3)))[:10]
    print(f"{'scheme':40s} {'mean deg':>9s} {'p99 deg':>9s} | worst-10 joints mean / p99")
    for k, v in e.items():
        print(f"{k:40s} {v.mean():9.3f} {np.percentile(v, 99):9.3f} | {v[:, worst].mean():8.3f} / {np.percentile(v[:, worst], 99):8.3f}")
    print("one code is", round(360 / 65536, 4), "degrees of twist")
