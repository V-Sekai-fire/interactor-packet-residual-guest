# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Ternary Gated DeltaNet-2 student. q, k, v and the two MLP head layers are ternary from the first step (absmean scale per block of 128,
fp16-rounded, straight-through). The state path stays higher precision, as Bonsai keeps ssm_alpha/ssm_beta in F32: the decay, erase and write
gate projections are float in training and 16-bit fixed point in the integer step. Inputs pad 180 -> 256 and the read-out pads 64 -> 128 to fit
blocks of 128. Dead-reckoning skip, L1 loss, 100-tick windows, same ticks and wire format as gdn3.py. Integer step checked against the float model."""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_gdn as g
import rung3 as r0
import gdn3

torch.set_num_threads(8)
T = os.environ["TMPDIR"].rstrip("/")
FX, BLOCK = 16, 128


class TGDN(nn.Module):
    def __init__(self, H=2, dk=32, dv=32, hid=256):
        super().__init__()
        self.H, self.dk, self.dv = H, dk, dv
        self.q, self.k, self.v = (r0.TLinear(256, H * d, True) for d in (dk, dk, dv))
        self.a, self.b, self.w = nn.Linear(256, H * dk), nn.Linear(256, H * dk), nn.Linear(256, H * dv)  # state path, not ternary
        nn.init.constant_(self.a.bias, 2.0)
        self.m1, self.m2 = r0.TLinear(128 + 256, hid, True), r0.TLinear(hid, 90, True)

    def forward(self, x):  # (B, L, 180) -> (B, L, 90), newest delta added back
        B, Lx, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv
        xp = F.pad(x, (0, 76))
        q = F.normalize(self.q(xp).view(B, Lx, H, dk), dim=-1, eps=1e-4)
        k = F.normalize(self.k(xp).view(B, Lx, H, dk), dim=-1, eps=1e-4)
        v = self.v(xp).view(B, Lx, H, dv)
        al = torch.sigmoid(self.a(xp)).view(B, Lx, H, dk)
        be = torch.sigmoid(self.b(xp)).view(B, Lx, H, dk)
        wr = torch.sigmoid(self.w(xp)).view(B, Lx, H, dv)
        St = torch.zeros(B, H, dv, dk)
        outs = []
        for t in range(Lx):
            Sd = St * al[:, t, :, None, :]
            r = torch.einsum("bhvk,bhk->bhv", Sd, be[:, t] * k[:, t])
            St = Sd + (wr[:, t] * v[:, t] - r)[..., None] * k[:, t][:, :, None, :]
            outs.append(torch.einsum("bhvk,bhk->bhv", St, q[:, t]).reshape(B, -1))
        o = F.pad(torch.stack(outs, 1), (0, 64))
        h = torch.relu(self.m1(torch.cat([o, xp], -1)))
        return self.m2(h) + x[:, :, :90]


def _tern(l):
    q, s = l.quant()
    return (q.view(l.o, l.i).numpy().astype(np.int8), np.rint(s[:, :, 0].numpy() * (1 << FX)).astype(np.int64),
            np.rint(l.b.detach().numpy().astype(np.float64) * (1 << FX)).astype(np.int64))


def _fix(l):
    f = lambda t: np.rint(t.detach().numpy().astype(np.float64) * (1 << FX)).astype(np.int64)
    return f(l.weight), f(l.bias)


class IntTGDN:
    def __init__(self, m):
        self.H, self.dk, self.dv = m.H, m.dk, m.dv
        with torch.no_grad():
            self.q, self.k, self.v, self.m1, self.m2 = (_tern(l) for l in (m.q, m.k, m.v, m.m1, m.m2))
            self.a, self.b, self.w = (_fix(l) for l in (m.a, m.b, m.w))

    @staticmethod
    def tlin(p, x):  # ternary layer, integer arithmetic only
        w, s, b = p
        acc = np.zeros((len(x), w.shape[0]), dtype=np.int64)
        for bl in range(w.shape[1] // BLOCK):
            acc += (x[:, bl * BLOCK:(bl + 1) * BLOCK] @ w[:, bl * BLOCK:(bl + 1) * BLOCK].astype(np.int64).T) * s[:, bl][None]
        return ((acc + (1 << (FX - 1))) >> FX) + b

    @staticmethod
    def flin(p, x):
        return ((x @ p[0].T + (1 << (FX - 1))) >> FX) + p[1]

    def step(self, St, x_codes):
        n, R = len(x_codes), 1 << (FX - 1)
        x = (x_codes.astype(np.int64) << FX) // 32768
        xp = np.concatenate([x, np.zeros((n, 76), dtype=np.int64)], 1)
        q = g.l2n_int(self.tlin(self.q, xp).reshape(n, self.H, self.dk))
        k = g.l2n_int(self.tlin(self.k, xp).reshape(n, self.H, self.dk))
        v = self.tlin(self.v, xp).reshape(n, self.H, self.dv)
        al = g.sig_int(self.flin(self.a, xp)).reshape(n, self.H, self.dk)
        be = g.sig_int(self.flin(self.b, xp)).reshape(n, self.H, self.dk)
        wr = g.sig_int(self.flin(self.w, xp)).reshape(n, self.H, self.dv)
        Sd = (St * al[:, :, None, :] + R) >> FX
        e = (be * k + R) >> FX
        r = ((Sd * e[:, :, None, :]).sum(-1) + R) >> FX
        z = (wr * v + R) >> FX
        St = Sd + (((z - r)[..., None] * k[:, :, None, :] + R) >> FX)
        o = ((St * q[:, :, None, :]).sum(-1) + R) >> FX
        o = np.concatenate([o.reshape(n, -1), np.zeros((n, 64), dtype=np.int64)], 1)
        h = np.maximum(self.tlin(self.m1, np.concatenate([o, xp], 1)), 0)
        return St, np.clip((self.tlin(self.m2, h) * 32768 + R) >> FX, -32767, 32767)


class _Wrap:  # gdn3.measure adds the skip itself; the model here already includes it, so the integer side returns the bare head output
    def __init__(self, m): self.i = IntTGDN(m); self.H, self.dv, self.dk = m.H, m.dv, m.dk
    def step(self, St, x): return self.i.step(St, x)


if __name__ == "__main__":
    # 1. integer step against the float model on an untrained net (planted control: a wiped state must differ)
    torch.manual_seed(1); m0 = TGDN(); rng = np.random.default_rng(0)
    C = np.cumsum(rng.integers(-300, 300, size=(2, 24, 90)), axis=1).astype(np.int32)
    xc = np.concatenate([g.wrap16(C[:, 1:] - C[:, :-1]), C[:, 1:]], 2)
    with torch.no_grad():
        pf = np.rint(m0(torch.tensor(xc / 32768.0, dtype=torch.float32)).numpy() * 32768)
    ig = IntTGDN(m0); st = np.zeros((2, 2, 32, 32), dtype=np.int64); st2 = st.copy(); w, bad = [], []
    for t in range(xc.shape[1]):
        st, pi = ig.step(st, xc[:, t]); w.append(int(np.abs(np.clip(pi + xc[:, t, :90], -32767, 32767) - pf[:, t]).max()))
        st2, pj = ig.step(st2, xc[:, t]); st2 = np.zeros_like(st2) if t % 2 == 0 else st2
        bad.append(int(np.abs(np.clip(pj + xc[:, t, :90], -32767, 32767) - pf[:, t]).max()))
    print(f"integer vs float, 23 ticks: max {max(w)} codes | CONTROL planted state wipe: max {max(bad)}", flush=True)
    # 2. train and measure
    testc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 90).astype(np.int32)
    X, Y = gdn3.windows(g.tick_seqs("train"), 16)
    dr = g.wire_cost([[g.stream(g.wrap16(testc[a, t + 1] - testc[a, t] - (testc[a, t] - testc[a, t - 1]))) for a in range(8)] for t in range(100, 398)])
    print(f"windows {tuple(X.shape)}; dead reckoning {dr:.0f} B/tick", flush=True)
    torch.manual_seed(11)
    m = gdn3.fit(TGDN(), X, Y, 24, "l1")
    gdn3.measure("ternary GDN-2 student, integer, l1 + skip, 100-tick state", _Wrap(m), testc)
    nz = [float(np.mean(p[0] != 0)) for p in (_tern(l) for l in (m.q, m.k, m.v, m.m1, m.m2))]
    print("ternary weights non-zero:", [f"{v*100:.0f}%" for v in nz])
