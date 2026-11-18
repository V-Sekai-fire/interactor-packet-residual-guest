# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Staged residual codes: a residual sent as stages of g payload bits plus a continue bit, stopping when nothing is left, against the current proxy cost
(0 for an exact zero, log2|r|+2 up to 126, 16 for an escape). Held-out avatars, ticks 101..398, dead-reckoning residuals. Proxy bits, not wire bytes."""
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("~/src/interactor-packet-residual-guest/train"))
import rot_frontier as r
T = r.T
tc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64)
c2, c1, c0 = tc[:, 99:397], tc[:, 100:398], tc[:, 101:399]
res = r.wrap16(c0 - r.p_codes(c2, c1)); a = np.abs(res); n = 8 * c0.shape[1]
cur = r.bits_of(a)
def staged(g, sign=True):
    """zero costs 0 (zero-run); otherwise ceil((bitlen + sign) / g) stages of g payload bits + 1 continue bit each"""
    bl = np.where(a == 0, 0, np.floor(np.log2(np.maximum(a, 1))).astype(int) + 1) + (1 if sign else 0)
    return np.where(a == 0, 0, np.ceil(bl / g) * (g + 1)).astype(float)
print(f"current proxy (<=126: log2|r|+2, else 16): {cur.sum()/n:6.0f} bits per avatar-tick")
for g in (2, 3, 4, 5, 6):
    b = staged(g); print(f"staged, {g} payload bits + 1 continue bit per stage: {b.sum()/n:6.0f}   (escapes {b[a>126].sum()/n:5.0f} of it, current {cur[a>126].sum()/n:5.0f})")
# the zero-run: the zero flag is free in the proxy for both; add one stage-less flag bit per nonzero for the staged code is already inside the stage count
# control: shuffle residual magnitudes across joints; a variable-length code must not beat the current by construction on the shuffled data
rng = np.random.default_rng(0); sh = rng.permutation(a.reshape(-1)).reshape(a.shape)
print("control, residuals shuffled (current / staged g=4):", round(float(r.bits_of(sh).sum()/n)), round(float(np.where(sh==0,0,np.ceil((np.floor(np.log2(np.maximum(sh,1)))+2)/4)*5).sum()/n)))
print()
big = a[a > 126]
print("escape residuals: n", big.size, "| percentiles 50/90/99 of |r|:", np.percentile(big, [50, 90, 99]).round(0), "| share above 1024:", round(float((big > 1024).mean()) * 100, 1), "%")
for g in (3, 4, 5):
    hyb = np.where(a > 126, np.minimum(16.0, staged(g)), cur)
    print(f"hybrid (<=126 as now; escapes the cheaper of 16 bits and staged g={g}): {hyb.sum()/n:6.0f} bits per avatar-tick  ({(hyb.sum()-cur.sum())/cur.sum()*100:+.1f}%)")
