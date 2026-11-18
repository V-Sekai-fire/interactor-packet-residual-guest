# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Predict from one cycle ago. Gait repeats, so a joint's value now is close to its value one period earlier (the idea under WalkTheDog's phase manifold, in its simplest
non-learned form). The period P is estimated per avatar and tick from past ticks only (the lowest mean code difference between the last K ticks and the K ticks P earlier, over all 90 codes),
so a decoder computes the same P. Predictors, all through the same size-class cost (tables fitted on four avatars, scored on the other four):
  dead reckoning      c[t-1] + (c[t-1] - c[t-2])
  cycle copy          c[t-P]
  cycle + step        c[t-1] + (c[t-P] - c[t-1-P])      the last tick's value moved by what the signal did one cycle ago
  best of the three, per joint-component, by the error over the last 4 ticks (decoder-computable).
Control: the period taken from a different avatar's history."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rot_frontier as r

tc = np.load(r.T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64)
PMIN, PMAX, K, START = 6, 70, 24, 101


def period(c, t):
    """c (T, 90) codes; the period chosen from ticks before t."""
    best, bp = None, PMIN
    for P in range(PMIN, PMAX + 1):
        if t - 1 - K - P < 0: break
        d = np.abs(r.wrap16(c[t - K:t] - c[t - K - P:t - P])).mean()
        if best is None or d < best: best, bp = d, P
    return bp


def predictions(c, periods):
    """c (T, 90); periods per tick. Returns dict of (T, 90) predictions for t in START..T-1."""
    T = len(c); out = {n: np.zeros_like(c) for n in ("dead reckoning", "cycle copy", "cycle + step")}
    for t in range(START, T):
        P = periods[t]
        out["dead reckoning"][t] = c[t - 1] + r.wrap16(c[t - 1] - c[t - 2])
        out["cycle copy"][t] = c[t - P]
        out["cycle + step"][t] = c[t - 1] + r.wrap16(c[t - P] - c[t - 1 - P])
    return out


def costs(res_by_name, tr, te):
    """size-class ideal bits per avatar-tick, per joint x component table fitted on tr avatars, scored on te"""
    out = {}
    for name, res in res_by_name.items():
        a = np.abs(res); cls = np.where(a == 0, 0, np.floor(np.log2(np.maximum(a, 1))).astype(int) + 1)
        cnt = np.full((90, 17), 0.05)
        for k in range(90): cnt[k] += np.bincount(cls[tr][:, :, k].ravel(), minlength=17)
        p = cnt / cnt.sum(-1, keepdims=True)
        b = -np.log2(p[np.arange(90)[None, None], cls[te]]) + np.where(cls[te] == 0, 0, cls[te])
        out[name] = b.sum() / (len(te) * cls.shape[1])
    return out


def main():
    C = tc.reshape(8, 400, 90)
    per = np.zeros((8, 400), int)
    for i in range(8):
        for t in range(START, 400): per[i, t] = period(C[i], t)
    print("estimated period (ticks), median per avatar:", [int(np.median(per[i, START:])) for i in range(8)], "\n")
    preds = [predictions(C[i], per[i]) for i in range(8)]
    names = ("dead reckoning", "cycle copy", "cycle + step")
    res = {n: np.stack([C[i, START:] - preds[i][n][START:] for i in range(8)]) for n in names}
    for n in names: res[n] = r.wrap16(res[n])
    # best of three per joint-component by the error over the last 4 ticks (past residuals only)
    abs_all = np.stack([np.abs(res[n]) for n in names])                  # (3, 8, T', 90)
    mix = np.zeros_like(res["cycle copy"])
    for t in range(res["cycle copy"].shape[1]):
        lo = max(0, t - 4)
        score = abs_all[:, :, lo:t].sum(2) if t > 0 else np.zeros_like(abs_all[:, :, 0])
        pick = score.argmin(0)                                            # (8, 90)
        mix[:, t] = np.take_along_axis(np.stack([res[n][:, t] for n in names]), pick[None], 0)[0]
    res["best of three (last 4 ticks)"] = mix
    A, B = [0, 1, 2, 3], [4, 5, 6, 7]
    c1, c2 = costs(res, A, B), costs(res, B, A)
    print(f"{'predictor':34s} {'bits / avatar-tick':>18s}  per test clip")
    for n in res: print(f"{n:34s} {(c1[n] + c2[n]) / 2:18.0f}")
    print("\nper clip (bits per tick; tables fitted on all eight avatars, so a little optimistic equally for every predictor):")
    def per_avatar(n):
        a_ = np.abs(res[n]); cls = np.where(a_ == 0, 0, np.floor(np.log2(np.maximum(a_, 1))).astype(int) + 1)
        cnt = np.full((90, 17), 0.05)
        for k in range(90): cnt[k] += np.bincount(cls[:, :, k].ravel(), minlength=17)
        p_ = cnt / cnt.sum(-1, keepdims=True)
        return (-np.log2(p_[np.arange(90)[None, None], cls]) + np.where(cls == 0, 0, cls)).sum(-1).mean(-1)
    pa = {n: per_avatar(n) for n in res}
    for i, name in enumerate(r.CLIPS):
        print(f"  {name:13s} period {int(np.median(per[i, START:])):3d} | dead reckoning {pa['dead reckoning'][i]:5.0f} | cycle copy {pa['cycle copy'][i]:5.0f} | cycle + step {pa['cycle + step'][i]:5.0f} | best of three {pa['best of three (last 4 ticks)'][i]:5.0f}")
    # control: periods from the wrong avatar
    wrong = np.roll(per, 1, axis=0)
    pw = [predictions(C[i], wrong[i]) for i in range(8)]
    rc = {"cycle copy, wrong avatar's period": r.wrap16(np.stack([C[i, START:] - pw[i]["cycle copy"][START:] for i in range(8)])),
          "cycle + step, wrong avatar's period": r.wrap16(np.stack([C[i, START:] - pw[i]["cycle + step"][START:] for i in range(8)]))}
    k1, k2 = costs(rc, A, B), costs(rc, B, A)
    print()
    for n in rc: print(f"control {n:36s} {(k1[n] + k2[n]) / 2:6.0f}")


if __name__ == "__main__":
    main()
