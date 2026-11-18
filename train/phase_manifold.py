# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""A learned phase manifold as a next-tick predictor (a small periodic autoencoder in the style of WalkTheDog / PAE, written from the equations in the paper).

Encoder: 1D convolutions over the last W ticks of 6D rotations (two matrix columns per joint, 180 values) -> D latent curves. Each curve gets an amplitude A, frequency f (from a DFT power
spectrum), offset B and phase s; the curve is rebuilt for W+1 ticks as A sin(2 pi (f t - s)) + B, so the last tick is an extrapolation. Decoder: 1D convolutions from the D curves back
to 180 values for the W+1 ticks. The decoded last tick is the prediction; it goes through Gram-Schmidt to a rotation and the packet's codes, and the residual is scored as in
cycle_predictor.py (size class per joint-component, tables fitted on four avatars and scored on the other four). The decoder predicts a correction on top of dead reckoning in the 6D space (trained with L1, with a learned dead zone so a still joint stays exactly still); the other window ticks are reconstructed relative to the last observed tick. Float model, float prediction (a measurement, not the shared integer decoder).
Training: the 66 library clips (test clips excluded) at the test tick spacing (every third 30 fps frame, three offsets). Control: the same model with its frequencies scrambled at test time."""
import glob, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rot_frontier as r
import one_euro_rot as oe
import cycle_predictor as cp

W, D, DT = 32, 8, 0.1
SCALE = 30.0
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def six(codes):  # (T,30,3) int codes -> (T,180) float: the first two columns of each joint's matrix
    return r.mats(codes)[..., :, :2].reshape(len(codes), 30 * 6).astype(np.float32)


class GDN2Enc(nn.Module):
    """Gated DeltaNet-2 (arXiv 2605.22791, clean-room from the equations) as the encoder: per tick, S^T_t = (S^T_{t-1} Dec_t) - ((S^T_{t-1} Dec_t)(b_t * k_t) - w_t * v_t) k_t^T with
    channel-wise decay, erase and write gates; o_t = S^T_t q_t; a per-tick linear reads the D latent values from o_t. (B, 180, W) -> (B, D, W)."""
    def __init__(self, H=2, dk=32, dv=32):
        super().__init__()
        self.H, self.dk, self.dv = H, dk, dv
        self.q, self.k, self.v = nn.Linear(180, H * dk), nn.Linear(180, H * dk), nn.Linear(180, H * dv)
        self.a, self.b, self.w = nn.Linear(180, H * dk), nn.Linear(180, H * dk), nn.Linear(180, H * dv)
        nn.init.constant_(self.a.bias, 2.0)
        self.out = nn.Linear(H * dv, D)

    def forward(self, x):
        x = x.transpose(1, 2); B, L, _ = x.shape; H, dk, dv = self.H, self.dk, self.dv
        q = nn.functional.normalize(self.q(x).view(B, L, H, dk), dim=-1, eps=1e-4)
        k = nn.functional.normalize(self.k(x).view(B, L, H, dk), dim=-1, eps=1e-4)
        v = self.v(x).view(B, L, H, dv)
        al = torch.sigmoid(self.a(x)).view(B, L, H, dk); be = torch.sigmoid(self.b(x)).view(B, L, H, dk); wr = torch.sigmoid(self.w(x)).view(B, L, H, dv)
        St = torch.zeros(B, H, dv, dk, device=x.device); outs = []
        for t in range(L):
            Sd = St * al[:, t, :, None, :]
            e = be[:, t] * k[:, t]
            rr = torch.einsum("bhvk,bhk->bhv", Sd, e)
            St = Sd + (wr[:, t] * v[:, t] - rr)[..., None] * k[:, t][:, :, None, :]
            outs.append(torch.einsum("bhvk,bhk->bhv", St, q[:, t]).reshape(B, -1))
        return self.out(torch.stack(outs, 1)).transpose(1, 2)


ENC = os.environ.get("PAE_ENC", "gdn2")


class PAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = GDN2Enc() if ENC == "gdn2" else nn.Sequential(nn.Conv1d(180, 96, 5, padding=2), nn.ELU(), nn.Conv1d(96, D, 5, padding=2))
        self.phase = nn.Linear(W, 2)
        self.dec = nn.Sequential(nn.Conv1d(D, 96, 5, padding=2), nn.ELU(), nn.Conv1d(96, 180, 5, padding=2))
        k = torch.arange(1, W // 2 + 1).float(); t = torch.arange(W).float() * DT
        self.register_buffer("fk", k / (W * DT))
        self.register_buffer("C", torch.cos(2 * np.pi * self.fk[:, None] * t[None]))
        self.register_buffer("S", torch.sin(2 * np.pi * self.fk[:, None] * t[None]))
        self.register_buffer("tt", torch.arange(W + 1).float() * DT)
        self.tau = nn.Parameter(torch.full((180,), float(os.environ.get("TAU0", "0.0001"))))

    def params(self, x):  # x (B, 180, W)
        z = self.enc(x)
        re, im = z @ self.C.T, z @ self.S.T
        p = re ** 2 + im ** 2
        f = (p * self.fk).sum(-1) / (p.sum(-1) + 1e-6)
        A = torch.sqrt(2 * p.sum(-1) + 1e-8) / W
        B = z.mean(-1)
        xy = self.phase(z)  # (B, D, 2)
        s = torch.atan2(xy[..., 1], xy[..., 0]) / (2 * np.pi)
        return f, A, B, s

    def forward(self, x, fscale=None):
        f, A, B, s = self.params(x)
        if fscale is not None: f = f * fscale
        curve = A[..., None] * torch.sin(2 * np.pi * (f[..., None] * self.tt - s[..., None])) + B[..., None]
        o = self.dec(curve)  # (B, 180, W+1): offsets from the last observed tick, scaled; the last entry is a correction to dead reckoning
        last = torch.sign(o[:, :, -1]) * torch.relu(o[:, :, -1].abs() - self.tau.abs() * SCALE)
        return torch.cat([o[:, :, :-1], last[:, :, None]], 2)


def load_train():
    names = set(r.CLIPS); seqs = []
    for f in sorted(glob.glob(r.T + "/more/*.npy")):
        if os.path.basename(f)[:-4] in names: continue
        C = np.load(f).astype(np.int64).reshape(-1, 30, 3); C[:, :, :2] *= -1
        for off in range(3):
            q = C[off::3]
            if len(q) > W + 5: seqs.append(six(q))
    return seqs


def rel_target(win):
    """win (B,180,W+1) -> targets: ticks 0..W-1 as offsets from tick W-1; the last entry is the second difference (the correction dead reckoning needs), all scaled."""
    ref = win[:, :, W - 1:W]
    rel = (win[:, :, :W] - ref) * SCALE
    corr = (win[:, :, W] - 2 * win[:, :, W - 1] + win[:, :, W - 2]) * SCALE
    return torch.cat([rel, corr[:, :, None]], 2)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    seqs = load_train(); lens = [len(s) for s in seqs]
    X = torch.tensor(np.concatenate(seqs)).to(DEV)
    starts = np.concatenate([np.arange(o, o + L - W) for o, L in zip(np.cumsum([0] + lens[:-1]), lens)])
    print(f"encoder {ENC}; {len(seqs)} sequences, {sum(lens)} ticks, {len(starts)} windows, device {DEV}", flush=True)
    m = PAE().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    idx_t = torch.tensor(starts).to(DEV); ar = torch.arange(W + 1).to(DEV)
    wt = torch.ones(W + 1).to(DEV); wt[-1] = float(os.environ.get("LASTW", "1000"))
    t0 = time.time()
    for it in range(steps):
        b = idx_t[torch.randint(len(idx_t), (256,), device=DEV)]
        win = X[b[:, None] + ar[None]].transpose(1, 2)                  # (B, 180, W+1)
        out = m(win[:, :, :W])
        tgt = rel_target(win)
        loss = ((out - tgt).abs().mean(1) * wt).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0 or it == steps - 1: print(f"step {it:5d} loss {loss.item():.5f}  {time.time() - t0:.0f}s", flush=True)
    m.eval()
    tc = np.load(r.T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64)
    START = 101
    def predict(fscale=None):
        P = np.zeros((8, 400, 90), np.int64)
        for i in range(8):
            xs = torch.tensor(six(tc[i])).to(DEV)
            wins = torch.stack([xs[t - W:t].T for t in range(START, 400)])  # (T', 180, W)
            with torch.no_grad(): corr = m(wins, None if fscale is None else fscale)[:, :, -1] / SCALE
            out = (2 * wins[:, :, -1] - wins[:, :, -2] + corr).cpu().numpy()  # dead reckoning in 6D plus the correction
            R = oe.six_to_rot(out.reshape(-1, 30, 3, 2).astype(np.float64))
            P[i, START:] = r.codes(R).reshape(-1, 90)
        return P
    C = tc.reshape(8, 400, 90)
    def dr6():
        P0 = np.zeros((8, 400, 90), np.int64)
        for i in range(8):
            x = six(tc[i]); R = oe.six_to_rot((2 * x[START - 1:-1] - x[START - 2:-2]).reshape(-1, 30, 3, 2).astype(np.float64)); P0[i, START:] = r.codes(R).reshape(-1, 90)
        return P0[:, START:]
    base = cp.predictions  # dead reckoning for the same ticks
    res = {}
    dr = np.stack([r.wrap16(C[i, START:] - (C[i, START - 1:-1] + r.wrap16(C[i, START - 1:-1] - C[i, START - 2:-2]))) for i in range(8)])
    res["dead reckoning"] = dr
    P = predict(); res["dead reckoning in 6D (no model)"] = r.wrap16(C[:, START:] - dr6()); res["learned phase manifold"] = r.wrap16(C[:, START:] - P[:, START:])
    rng = np.random.default_rng(5); fs = torch.tensor(rng.uniform(0.5, 2.0, (1, D)).astype(np.float32)).to(DEV)
    P2 = predict(fs); res["control, frequencies scrambled"] = r.wrap16(C[:, START:] - P2[:, START:])
    # control: predict "no change" (copy the last tick) for scale
    res["copy the last tick"] = np.stack([r.wrap16(C[i, START:] - C[i, START - 1:-1]) for i in range(8)])
    A, B = [0, 1, 2, 3], [4, 5, 6, 7]
    c1, c2 = cp.costs(res, A, B), cp.costs(res, B, A)
    print("\nbits per avatar-tick (size class per joint-component, tables fitted on four avatars, scored on the other four):")
    for n in res: print(f"  {n:34s} {(c1[n] + c2[n]) / 2:6.0f}   exact zero {np.mean(res[n] == 0) * 100:4.1f}%  escape {np.mean(np.abs(res[n]) > 126) * 100:4.1f}%")
    f, Aa, Bb, s = m.params(torch.tensor(six(tc[0])[START - W:START].T[None]).to(DEV))
    print("example latent frequencies (Hz) for walk_1:", np.round(f[0].detach().cpu().numpy(), 2))


if __name__ == "__main__":
    main()
