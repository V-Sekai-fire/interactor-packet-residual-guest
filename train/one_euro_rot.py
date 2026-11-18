# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Making the measured values the same: a One Euro filter and a hold band on tracked rotations, so slow jitter becomes exact repeats (zero residual bytes).

Tracked joints are the head and both hands. The test avatars are clean clip playback, so tracker noise is ADDED here: independent rotation jitter of sigma degrees per tick
(an assumption, swept, not a measurement of any tracker). Variants: the raw noisy measurement; a One Euro filter on SO(3) (slerp with an adaptive gain; the speed and
the cutoff follow frame-eye-osc's One Euro); a hold band (send a new code only when a component has moved more than h codes from the last sent one); both.
Each is scored two ways: ideal bits per avatar-tick for the 3 joints (size class per joint x component, tables fitted on four avatars, scored on the other four), and the
error against the CLEAN motion (degrees), because a filter that holds still is also a filter that lags."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rot_frontier as r

JOINTS = [6, 13, 19]  # Head, LeftHand, RightHand
DT = 1 / 20.0


def noisy(R, sigma_deg, seed):
    w = np.random.default_rng(seed).standard_normal(R.shape[:-2] + (3,)) * np.radians(sigma_deg)
    return R @ r.exp_so3(w)


def one_euro(Rm, fc_min, beta, dcut=1.0):
    """Rm (A, T, J, 3, 3) -> filtered. gain alpha = 1 / (1 + tau / dt), tau = 1 / (2 pi fc), fc = fc_min + beta * smoothed speed (rad/s)."""
    out = np.empty_like(Rm); out[:, 0] = Rm[:, 0]; dx = np.zeros(Rm.shape[:1] + Rm.shape[2:3])
    a_d = 1.0 / (1.0 + (1.0 / (2 * np.pi * dcut)) / DT)
    for t in range(1, Rm.shape[1]):
        step = r.log_so3(r.mT(out[:, t - 1]) @ Rm[:, t])
        speed = np.linalg.norm(step, axis=-1) / DT
        dx = dx + a_d * (speed - dx)
        fc = fc_min + beta * dx
        a = 1.0 / (1.0 + (1.0 / (2 * np.pi * fc)) / DT)
        out[:, t] = out[:, t - 1] @ r.exp_so3(a[..., None] * step)
    return out


def six_to_rot(s6):
    """(…, 3, 2) two columns -> rotation by Gram-Schmidt (the 6D representation: continuous, no angle to wrap)."""
    a, b = s6[..., 0], s6[..., 1]
    c0 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (c0 * b).sum(-1, keepdims=True) * c0
    c1 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([c0, c1, np.cross(c0, c1)], -1)


def one_euro6(Rm, fc_min, beta, dcut=1.0):
    """The same gain rule, but the state is the 6D form and the blend is a straight line in it."""
    out = np.empty_like(Rm); out[:, 0] = Rm[:, 0]; s6 = Rm[:, 0][..., :2].copy(); dx = np.zeros(Rm.shape[:1] + Rm.shape[2:3])
    a_d = 1.0 / (1.0 + (1.0 / (2 * np.pi * dcut)) / DT)
    for t in range(1, Rm.shape[1]):
        speed = np.linalg.norm(r.log_so3(r.mT(out[:, t - 1]) @ Rm[:, t]), axis=-1) / DT
        dx = dx + a_d * (speed - dx)
        a = 1.0 / (1.0 + (1.0 / (2 * np.pi * (fc_min + beta * dx))) / DT)
        s6 = s6 + a[..., None, None] * (Rm[:, t][..., :2] - s6)
        out[:, t] = six_to_rot(s6)
    return out


def one_euro_innovation(Rm, fc_min, beta, dcut=1.0):
    """The rhythm-game structure: a smooth reference plus a filtered delta. The reference is the constant-velocity prediction from the last two outputs; the 1 Euro filter
    runs on the innovation log(pred^T measured), and the output is pred * exp(filtered innovation)."""
    out = np.empty_like(Rm); out[:, :2] = Rm[:, :2]
    ef = np.zeros(Rm.shape[:1] + Rm.shape[2:3] + (3,)); dx = np.zeros_like(ef)
    a_d = 1.0 / (1.0 + (1.0 / (2 * np.pi * dcut)) / DT)
    for t in range(2, Rm.shape[1]):
        pred = out[:, t - 1] @ r.exp_so3(r.log_so3(r.mT(out[:, t - 2]) @ out[:, t - 1]))
        e = r.log_so3(r.mT(pred) @ Rm[:, t])
        d = (e - ef) / DT
        dx = dx + a_d * (d - dx)
        a = 1.0 / (1.0 + (1.0 / (2 * np.pi * (fc_min + beta * np.linalg.norm(dx, axis=-1)))) / DT)
        ef = ef + a[..., None] * (e - ef)
        out[:, t] = pred @ r.exp_so3(ef)
        ef = r.log_so3(r.mT(pred) @ out[:, t])   # the filtered innovation as it was applied
    return out


def hold(C, h):
    """C (A, T, J, 3) int codes: a component keeps its last sent value until the new one is more than h codes away (twist compared on the circle)."""
    out = C.copy()
    for t in range(1, C.shape[1]):
        d = r.wrap16(C[:, t] - out[:, t - 1])
        out[:, t] = np.where(np.abs(d) > h, C[:, t], out[:, t - 1])
    return out


def bits(C, tr, te, pred):
    """C (A, T, 3, 3): ideal bits per avatar-tick of the dead-reckoning (or hold) residuals."""
    c2, c1, c0 = C[:, :-2], C[:, 1:-1], C[:, 2:]
    if pred == "dead reckoning": p = r.p_codes(c2, c1)
    elif pred == "6D": p = r.codes(six_to_rot(2 * r.mats(c1)[..., :2] - r.mats(c2)[..., :2]))
    else: p = c1
    res = np.abs(r.wrap16(c0 - p))
    cls = np.where(res == 0, 0, np.floor(np.log2(np.maximum(res, 1))).astype(int) + 1)
    cnt = np.full((3, 3, 17), 0.05)
    for j in range(3):
        for k in range(3):
            cnt[j, k] += np.bincount(cls[tr][:, :, j, k].ravel(), minlength=17)
    p = cnt / cnt.sum(-1, keepdims=True)
    jj, kk = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    b = -np.log2(p[jj[None, None], kk[None, None], cls[te]]) + np.where(cls[te] == 0, 0, cls[te])
    return b.sum() / (len(te) * cls.shape[1]), float(np.mean(res == 0)) * 100


def score(C, pred):
    A, B = [0, 1, 2, 3], [4, 5, 6, 7]
    (b1, z1), (b2, z2) = bits(C, A, B, pred), bits(C, B, A, pred)
    return (b1 + b2) / 2, (z1 + z2) / 2


def main():
    tc = np.load(r.T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64)[:, :, JOINTS]
    clean = r.mats(tc)
    print("bits per avatar-tick for head + both hands (3 joints x 3 codes); dead reckoning unless noted; error against the clean motion in degrees (mean / p99)\n")
    for sigma in (0.1, 0.3):
        Rm = noisy(clean, sigma, 11)
        variants = {"raw noisy measurement": r.codes(Rm)}
        for fc, beta in ((1.0, 0.5), (2.0, 2.0)):
            variants[f"One Euro fc_min {fc} beta {beta}"] = r.codes(one_euro(Rm, fc, beta))
        for fc, beta in ((1.0, 0.5), (2.0, 2.0)):
            variants[f"One Euro on the innovation, fc_min {fc} beta {beta}"] = r.codes(one_euro_innovation(Rm, fc, beta))
        variants["One Euro in 6D, fc_min 1.0 beta 0.5"] = r.codes(one_euro6(Rm, 1.0, 0.5))
        for h in (8, 32):
            variants[f"hold band {h} codes"] = hold(r.codes(Rm), h)
        variants["One Euro fc_min 1.0 beta 0.5 + hold 8"] = hold(r.codes(one_euro(Rm, 1.0, 0.5)), 8)
        print(f"jitter sigma {sigma} deg per tick")
        print(f"  {'variant':40s} {'bits':>6s} {'exact zero':>10s} {'hold-last':>9s} {'6D pred':>8s} {'error mean':>10s} {'p99':>6s}")
        for name, C in variants.items():
            b, z = score(C, "dead reckoning"); bh, _ = score(C, "hold"); b6, _ = score(C, "6D")
            e = r.mT(clean) @ r.mats(C); ang = np.degrees(np.linalg.norm(r.log_so3(e), axis=-1))
            print(f"  {name:40s} {b:6.0f} {z:9.1f}% {bh:9.0f} {b6:8.0f} {ang.mean():10.3f} {np.percentile(ang, 99):6.2f}")
    b, z = score(r.codes(clean), "dead reckoning"); print(f"\nclean clip playback (no jitter), dead reckoning: {b:.0f} bits, exact zero {z:.1f}%")


if __name__ == "__main__":
    main()
