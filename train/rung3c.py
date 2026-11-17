# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Rung 3c: encoder-side learned model, integer decoder. The encoder sees the true next delta and picks a short code z
(d digits of 5 levels per avatar-frame); the decoder is an integer function of (its own history, z) and adds the residual.
The encoder's float noise never has to match the decoder, because the decoder is defined by its integer layers. Every packet is
decoded back and compared; the cost counts the code z and the residual."""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_gdn as g
import rung3b as r

torch.set_num_threads(4)
T = os.environ["TMPDIR"].rstrip("/")
SC, PIN, L = 32768.0, r.PIN, 5
HALF = (L - 1) // 2


class CVAE(nn.Module):
    def __init__(self, d, ternary_dec, h=256):
        super().__init__()
        self.d = d
        self.enc = nn.Sequential(nn.Linear(PIN + 90, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d))  # float, encoder side only
        self.dec = r.MLP(ternary_dec, h, skip=True)  # integer-exportable decoder; its input holds the history and the code

    def code(self, ctx, y):
        z = torch.tanh(self.enc(torch.cat([ctx, y], 1))) * HALF
        return z + (torch.round(z) - z).detach()

    def forward(self, ctx, y):
        z = self.code(ctx, y)
        x = ctx.clone()
        x[:, 180:180 + self.d] = z / HALF
        return self.dec(x), z


def fit(m, X, Y, epochs, lr=2e-3, bs=512, log=""):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m.to(dev); X, Y = X.to(dev), Y.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=dev); tot = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            p, _ = m(X[idx], Y[idx])
            loss = (p - Y[idx]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
        sch.step()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  {log} epoch {ep+1}/{epochs} L1 {tot/len(X):.3e}", flush=True)
    return m.cpu()


def evaluate(name, m, d, testc):
    """Encoder in float on the CPU picks z for each avatar-frame; the integer decoder predicts; the residual fixes the rest."""
    lt = r.int_layers(m.dec)
    blobs, ab, ok, zb = [], [], True, 0
    for t in range(8, 399):
        x = np.zeros((8, PIN), dtype=np.int64)
        x[:, :90] = g.wrap16(testc[:, t] - testc[:, t - 1]); x[:, 90:180] = testc[:, t]
        true = g.wrap16(testc[:, t + 1] - testc[:, t])
        with torch.no_grad():
            z = np.rint(m.code(torch.tensor(x / SC, dtype=torch.float32), torch.tensor(true / SC, dtype=torch.float32)).numpy()).astype(np.int64)
        # the decoder, from integers only: its history x and the digits z
        xd = x * 2
        xd[:, 180:180 + d] = z * (1 << 16) // HALF
        pred = np.clip(r.run_int(lt, xd) + x[:, :90], -32767, 32767)
        res = g.wrap16(true - pred)
        back = g.wrap16(pred + res)
        ok &= bool(np.array_equal(back, true))
        ab.append(np.abs(res))
        blobs.append([(z[a] + HALF).astype(np.uint8).tobytes() + g.stream(res[a]) for a in range(8)])
    c = g.wire_cost(blobs); e = np.concatenate(ab).ravel()
    print(f"TEST {name:46s} {c:6.0f} B/tick = {c*8/240:5.1f} bits/joint | residual zero {np.mean(e==0)*100:4.1f}%, median {np.median(e):.0f}, escape {np.mean(e>126)*100:4.1f}% | decode exact: {ok}", flush=True)


if __name__ == "__main__":
    testc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 90).astype(np.int32)
    X, Y = r.pairs(g.tick_seqs("train"))
    print("dead reckoning (last delta): 1067 B/tick from rung 3 on the same ticks", flush=True)
    for d in (8, 16, 32):
        for tern in (False, True):
            torch.manual_seed(11)
            m = fit(CVAE(d, tern), X, Y, 60, log=f"d={d} {'ternary' if tern else 'float'} decoder")
            evaluate(f"code d={d} digits of {L} levels, {'ternary' if tern else '16-bit'} integer decoder", m, d, testc)
