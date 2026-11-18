# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""A residual FSQ with a hand-computed codebook, no network. The dead-reckoning residual of one avatar-tick is a 90-vector (30 joints x swing_x, swing_z, twist).
Transform: the principal axes of the training residuals (an SVD, closed form). Stages: each kept coefficient is rounded to one of L levels at step d_s, the next stage
rounds what is left at d_s / L, and a final integer fix brings back the residual exactly. The decoder rebuilds the estimate from the digits with fixed arithmetic.
Training residuals come from library clips (resampled to 20 Hz, test clips excluded); scoring is on the 8 held-out avatars. Proxy bits: digits and fix are coded with static
tables fitted on the training residuals (ideal code length)."""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rot_frontier as r

T = r.T


def to20(C):
    R = r.mats(C); n = len(C); t = np.arange(0, (n - 1) / 30.0, 1 / 20.0) * 30.0
    i = np.floor(t).astype(int); f = (t - i)[:, None]
    Ri, Rj = R[i], R[np.minimum(i + 1, n - 1)]
    return r.codes(Ri @ r.exp_so3(f[..., None] * r.log_so3(r.mT(Ri) @ Rj)))


def residuals(C):  # (T,30,3) codes -> dead-reckoning residuals (T-2,30,3)
    return r.wrap16(C[2:] - r.p_codes(C[:-2], C[1:-1]))


def size_cost(X, tr, te):
    """ideal bits of integer array X (N,30,3): size class coded with a per joint x component table fitted on rows tr, low bits and sign raw; scored on rows te"""
    a = np.abs(X); cls = np.where(a == 0, 0, np.floor(np.log2(np.maximum(a, 1))).astype(int) + 1)
    cnt = np.full((30, 3, 17), 0.05)
    for j in range(30):
        for k in range(3): cnt[j, k] += np.bincount(cls[tr][:, j, k], minlength=17)
    p = cnt / cnt.sum(-1, keepdims=True)
    jj, kk = np.meshgrid(np.arange(30), np.arange(3), indexing="ij")
    bits = -np.log2(p[jj[None], kk[None], cls[te]]) + np.where(cls[te] == 0, 0, cls[te])
    return bits.sum() / len(te)


def sym_cost(D, tr, te, nsym):
    """ideal bits of small digit array D (N,m) with a per-column table fitted on rows tr"""
    bits = 0.0
    for c in range(D.shape[1]):
        cnt = np.full(nsym, 0.05) + np.bincount(D[tr][:, c], minlength=nsym)
        p = cnt / cnt.sum()
        bits += -np.log2(p[D[te][:, c]]).sum()
    return bits / len(te)


def main():
    test_names = set(r.CLIPS)
    train = []
    for f in sorted(glob.glob(T + "/more/*.npy")):
        if os.path.basename(f)[:-4] in test_names: continue
        C = np.load(f).astype(np.int64).reshape(-1, 30, 3); C[:, :, :2] *= -1
        if len(C) > 150: train.append(residuals(to20(C)))
    Rtr = np.concatenate(train).reshape(-1, 90).astype(float)
    tc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64)
    Rte = np.concatenate([r.wrap16(tc[i, 101:400] - r.p_codes(tc[i, 99:398], tc[i, 100:399])) for i in range(8)]).reshape(-1, 90)
    print(f"training residual vectors {len(Rtr)}, test {len(Rte)} (8 avatars x {len(Rte)//8} ticks)")
    clip = 2048.0
    Z = np.clip(Rtr, -clip, clip); mu = Z.mean(0)
    U, S, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    ev = S ** 2 / (len(Z) - 1)
    print("variance share of the top 1, 4, 8, 16, 32 principal axes:", [round(float(ev[:k].sum() / ev.sum()), 3) for k in (1, 4, 8, 16, 32)])
    print("transform gain (arithmetic / geometric mean of the axis variances): %.2f" % float(ev.mean() / np.exp(np.log(ev + 1e-9).mean())))
    # control: the same spectrum on residuals with each joint-component shuffled in time loses the structure
    sh = Z.copy(); rng = np.random.default_rng(0)
    for c in range(90): sh[:, c] = rng.permutation(sh[:, c])
    ev2 = np.linalg.svd(sh - sh.mean(0), compute_uv=False) ** 2 / (len(sh) - 1)
    print("control, each column shuffled in time: gain %.2f, top-8 share %.3f" % (float(ev2.mean() / np.exp(np.log(ev2 + 1e-9).mean())), float(ev2[:8].sum() / ev2.sum())))
    # baseline: the residual as it is, size class per joint x component (tables fitted on the training residuals)
    Xtr, Xte = Rtr.astype(np.int64).reshape(-1, 30, 3), Rte.reshape(-1, 30, 3)
    n_tr = len(Xtr)
    both = np.concatenate([Xtr, Xte]); tr_idx = np.arange(n_tr); te_idx = np.arange(n_tr, len(both))
    base = size_cost(both, tr_idx, te_idx)
    print(f"\nbaseline, residual coded as it is (size class per joint-component): {base:.0f} bits per avatar-tick")
    print(f"{'kept axes':>9s} {'step/sd':>7s} {'levels':>6s} {'digit bits':>11s} {'fix bits':>9s} {'total':>7s} {'vs base':>8s}")
    allZ = np.concatenate([Xtr.reshape(-1, 90), Xte.reshape(-1, 90)]).astype(float)
    best = None
    rngc = np.random.default_rng(1); Q, _ = np.linalg.qr(rngc.standard_normal((90, 90)))  # planted wrong axes: a random orthogonal basis
    def run(Vfull, K, m, L, quiet=False):
        V = Vfull[:K]; coef = (np.clip(allZ, -clip, clip) - mu) @ V.T; sd = (coef[:n_tr]).std(0)
        d = m * sd; q = np.clip(np.round(coef / d), -(L // 2), L // 2); est = q * d
        recon = np.round(est @ V + mu).astype(np.int64).reshape(-1, 30, 3)
        fix = both - recon
        D = (q + L // 2).astype(int)
        return sym_cost(D, tr_idx, te_idx, L), size_cost(fix, tr_idx, te_idx)
    for K in (8, 32, 90):
        for m in (0.25, 0.5, 1.0, 2.0):
            for L in (5, 9, 17):
                dbits, fbits = run(Vt, K, m, L); tot = dbits + fbits
                print(f"{K:9d} {m:7.2f} {L:6d} {dbits:11.0f} {fbits:9.0f} {tot:7.0f} {(tot - base) / base * 100:+7.1f}%")
                if best is None or tot < best[0]: best = (tot, K, m, L)
    _, K, m, L = best
    dc, fc = run(Q, K, m, L)
    print(f"control, same grid on a random orthogonal basis instead of the principal axes: {dc + fc:.0f} bits")
    print("best:", best)


if __name__ == "__main__":
    main()
