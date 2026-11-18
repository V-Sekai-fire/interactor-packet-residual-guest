# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""A learned tokenizer as robust residual FSQ, in the packet's lossless frame. The sender sees a block of 4 ticks (the MotionBricks block size); a teacher MLP turns the block's change from the
last tick into D values, which residual FSQ quantises in S stages of L levels (stage s has step 1/L^(s-1); every stage is a digit in 0..L-1, so dropping the fine stages still decodes). The receiver's
predictor is the phase_manifold conv VAE (window of past ticks, the true earlier ticks of the block included) with the dequantised token as an extra input to every tick of the block. The residual to the
real codes stays lossless. Cost per avatar-tick = residual proxy bits (as in phase_manifold.py) + token bits / 4, the token bits from the empirical digit histograms of the training blocks (add-1 smoothing).
Controls: the token removed (zeros); another avatar's token; a model trained with no token (NOTOK=1). Float model; a measurement."""
import os, sys
os.environ.update(VAE_BETA="0", PAE_ENC="conv", PAE_IN="six")
DZ, S, L = int(os.environ.get("DZ", "8")), int(os.environ.get("S", "2")), int(os.environ.get("L", "5"))
os.environ["PAE_EXTRA"] = str(DZ)
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_manifold as pm
import rot_frontier as r
import one_euro_rot as oe
import cycle_predictor as cp

W, DEV, SCALE = pm.W, pm.DEV, pm.SCALE
NOTOK = os.environ.get("NOTOK", "0") == "1"; STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
H = (L - 1) / 2; torch.manual_seed(0); np.random.seed(0)


def rfsq(raw):
    """bounded value -> (dequantised token, digits (B, S, DZ)); stage s step is L^-s, digits 0..L-1"""
    z = torch.tanh(raw) * H; res = z; q = torch.zeros_like(z); digs = []
    for s in range(S):
        step = float(L) ** (-s); d = torch.clamp(torch.round(res / step), -H, H)
        qs = d * step; q = q + qs; res = z - q.detach() if False else res - qs; digs.append((d + H).long())
    return z + (q - z).detach(), torch.stack(digs, 1)  # straight-through to the teacher


class Teacher(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(4 * 180, 128), nn.ELU(), nn.Linear(128, DZ))
    def forward(self, blk): return self.net(blk.reshape(len(blk), -1) * SCALE)  # blk (B, 4, 180): the block's ticks minus the last context tick


seqs = pm.load_train(); lens = [len(q[0]) for q in seqs]
X = torch.tensor(np.concatenate([q[0] for q in seqs])).to(DEV)
offs = np.cumsum([0] + lens[:-1]); starts = np.concatenate([np.arange(o + W + 2, o + n - 4) for o, n in zip(offs, lens)])
B_all = torch.tensor(starts).to(DEV); ar = torch.arange(W + 1).to(DEV); a4 = torch.arange(4).to(DEV)
m = pm.PAE().to(DEV); tch = Teacher().to(DEV)
opt = torch.optim.Adam(list(m.parameters()) + list(tch.parameters()), 1e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
wt = torch.ones(W + 1).to(DEV); wt[-1] = pm.float(os.environ.get("LASTW", "1000")) if hasattr(pm, "float") else float(os.environ.get("LASTW", "1000"))
print(f"{len(starts)} blocks; token {DZ} dims x {S} stages x {L} levels = {DZ * S * np.log2(L):.1f} bits per block ({DZ * S * np.log2(L) / 4:.1f} per avatar-tick, before entropy coding); {'NO TOKEN' if NOTOK else ''}", flush=True)
m.train(); tch.train()
for it in range(STEPS):
    b = B_all[torch.randint(len(B_all), (256,), device=DEV)]
    zq = torch.zeros(256, DZ, device=DEV) if NOTOK else rfsq(tch(X[b[:, None] + a4[None]] - X[b - 1][:, None]))[0]
    loss = 0
    for k in range(4):
        win = X[(b + k - W)[:, None] + ar[None]].transpose(1, 2)
        out = m(win[:, :, :W], None, None, extra=zq)
        loss = loss + ((out - pm.rel_target(win)).abs().mean(1) * wt).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 500 == 0 or it == STEPS - 1: print(f"step {it:5d} loss {loss.item():.4f}", flush=True)
m.eval(); tch.eval()
hist = torch.ones(S, DZ, L, device=DEV)  # digit histograms on training blocks, add-1
with torch.no_grad():
    for _ in range(40):
        b = B_all[torch.randint(len(B_all), (512,), device=DEV)]; dg = rfsq(tch(X[b[:, None] + a4[None]] - X[b - 1][:, None]))[1]
        for s in range(S):
            for d in range(DZ): hist[s, d] += torch.bincount(dg[:, s, d], minlength=L).float()
lpt = torch.log2(hist / hist.sum(-1, keepdim=True))
tc = np.load(pm.r.T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64); C = tc.reshape(8, 400, 90)
bs = np.arange(101, 397, 4); bt = torch.tensor(bs).to(DEV); T0, T1 = 101, 397
xs_all = [torch.tensor(pm.six(tc[i])).to(DEV) for i in range(8)]
zs, digs = [], []
with torch.no_grad():
    for i in range(8):
        z, dg = rfsq(tch(torch.stack([xs_all[i][bt + j] - xs_all[i][bt - 1] for j in range(4)], 1)))
        zs.append(z); digs.append(dg)
    tok_bits = np.mean([(-lpt[torch.arange(S)[:, None], torch.arange(DZ)[None], dg].sum((1, 2))).mean().item() for dg in digs]) if not NOTOK else 0.0


def predict(tokens):
    Pc = np.zeros((8, 400, 90), np.int64)
    for i in range(8):
        xs = xs_all[i]
        for k in range(4):
            win = torch.stack([xs[t + k - W:t + k].T for t in bs])  # (blocks, 180, W)
            with torch.no_grad(): corr = m(win, None, None, extra=tokens[i])[:, :, -1] / SCALE
            pred6 = (2 * xs[bt + k - 1] - xs[bt + k - 2] + corr).cpu().numpy()
            Pc[i, bs + k] = r.codes(oe.six_to_rot(pred6.reshape(-1, 30, 3, 2).astype(np.float64))).reshape(-1, 90)
    return r.wrap16(C[:, T0:T1] - Pc[:, T0:T1])


res = {}
dr = np.stack([r.wrap16(C[i, T0:T1] - (C[i, T0 - 1:T1 - 1] + r.wrap16(C[i, T0 - 1:T1 - 1] - C[i, T0 - 2:T1 - 2]))) for i in range(8)])
res["dead reckoning"] = dr
res["conv predictor, token as sent"] = predict(zs)
if not NOTOK:
    res["control: token removed (zeros)"] = predict([torch.zeros_like(z) for z in zs])
    res["control: another avatar's token"] = predict([zs[(i + 1) % 8] for i in range(8)])
A, B2 = [0, 1, 2, 3], [4, 5, 6, 7]; c1, c2 = cp.costs(res, A, B2), cp.costs(res, B2, A)
print(f"\nbits per avatar-tick, ticks {T0}-{T1 - 1} (residual proxy; token bits added in the last column):")
for n in res:
    v = (c1[n] + c2[n]) / 2; extra = tok_bits / 4 if n == "conv predictor, token as sent" else 0
    print(f"  {n:36s} {v:6.0f}   escape {np.mean(np.abs(res[n]) > 126) * 100:4.1f}%   + token {extra:4.1f} = {v + extra:6.0f}")
print(f"token: {tok_bits:.1f} bits per block of 4 ticks after entropy coding ({tok_bits / 4:.1f} per avatar-tick); nominal {DZ * S * np.log2(L):.1f}")
