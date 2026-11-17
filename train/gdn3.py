# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Gated DeltaNet-2 student with the dead-reckoning skip, trained with a loss that counts exact hits (L1, and a soft-miss count), over 100-tick
windows (the client's 5 s receive timeout at 20 Hz). The state carries the history; the decoder runs the integer step on its own packets.
Same ticks (101..398) and wire format as rung3f, dead reckoning 1080."""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_gdn as g

torch.set_num_threads(8)
T = os.environ["TMPDIR"].rstrip("/")
SC, L, WARM = 32768.0, 100, 8


class GDNS(g.GDN):  # the same layer, the output added to the newest delta (dead reckoning)
    def forward(self, x):
        return super().forward(x) + x[:, :, :90]


def windows(seqs, stride):
    X, Y = [], []
    for C in seqs:
        for b in range(0, len(C) - L - 2, stride):
            seg = C[b:b + L + 2]
            d = g.wrap16(seg[1:] - seg[:-1])
            X.append(np.concatenate([d[:L], seg[1:L + 1]], axis=1)); Y.append(d[1:L + 1])
    return torch.tensor(np.array(X) / SC, dtype=torch.float32), torch.tensor(np.array(Y) / SC, dtype=torch.float32)


def fit(m, X, Y, epochs, mode, tau=64.0, bs=64, lr=2e-3):
    opt = torch.optim.Adam(m.parameters(), lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(X)); tot = 0.0; t0 = time.time()
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            e = (m(X[idx])[:, WARM:] - Y[idx][:, WARM:])
            loss = e.abs().mean() if mode == "l1" else torch.log1p((e * SC / tau) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
        sch.step()
        if ep % 3 == 0 or ep == epochs - 1:
            print(f"  {mode} epoch {ep+1}/{epochs} loss {tot/len(X):.3e} ({time.time()-t0:.0f}s/epoch)", flush=True)
    return m


def measure(name, m_int, testc, t0=101):
    blobs, ab, ok = [], [], True
    st = np.zeros((8, m_int.H, m_int.dv, m_int.dk), dtype=np.int64)
    for t in range(1, 399):
        x = np.concatenate([g.wrap16(testc[:, t] - testc[:, t - 1]), testc[:, t]], 1)
        st, out = m_int.step(st, x)  # the state follows the exact decoded packets
        if t >= t0 - 1 and t + 1 <= 398 + 1 and t + 1 >= t0:
            pred = np.clip(out + x[:, :90], -32767, 32767)  # the skip
            true = g.wrap16(testc[:, t + 1] - testc[:, t]); res = g.wrap16(true - pred)
            ok &= bool(np.array_equal(g.wrap16(pred + res), true))
            ab.append(np.abs(res)); blobs.append([g.stream(res[a]) for a in range(8)])
    c = g.wire_cost(blobs); e = np.concatenate(ab).ravel()
    print(f"TEST {name:46s} {c:6.0f} B/tick = {c*8/240:5.1f} bits/joint | zero {np.mean(e==0)*100:4.1f}%, median {np.median(e):.0f}, escape {np.mean(e>126)*100:4.1f}% | exact {ok}", flush=True)
    return c


if __name__ == "__main__":
    testc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 90).astype(np.int32)
    X, Y = windows(g.tick_seqs("train"), 16)
    print("windows", tuple(X.shape), flush=True)
    dr = [[g.stream(g.wrap16(testc[a, t + 1] - testc[a, t] - (testc[a, t] - testc[a, t - 1]))) for a in range(8)] for t in range(100, 398)]
    print(f"dead reckoning (same ticks): {g.wire_cost(dr):.0f} B/tick", flush=True)
    for mode in ("l1", "soft"):
        torch.manual_seed(11)
        m = fit(GDNS(), X, Y, 24, mode)
        measure(f"GDN-2 student, integer, {mode} + skip, 100-tick state", g.IntGDN(m), testc)
