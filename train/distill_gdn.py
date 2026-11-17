# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Gated DeltaNet-2 as the rotation predictor of the lossless packet codec, teacher and student, on the dataset's splits.
  S_t = a_t * S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T ,  o_t = S_t q_t     (q, k L2-normalised; a, b sigmoid gates)
The state is per avatar and is updated from exact decoded packets, so both ends hold the same state. The student runs in
integer arithmetic (2^16 fixed point, a sigmoid table, an integer square root); the teacher only trains."""
import math, os, sys, time
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard as zstd

T = os.environ["TMPDIR"].rstrip("/")
S = "/tmp/datasource-packet-residual-stage/data"
torch.manual_seed(7)
np.random.seed(7)
SC, FX, L = 32768.0, 16, 32  # code scale, fixed point bits, training window of ticks
T0 = 8  # every method is measured from the same tick


def wrap16(x):
    return ((x + 32768) % 65536) - 32768


def tick_seqs(split):
    t = pq.read_table(f"{S}/{split}.parquet", columns=["clip", "frame", "swing_twist"]).to_pydict()
    by = {}
    for c, f, st in zip(t["clip"], t["frame"], t["swing_twist"]):
        by.setdefault(c, []).append((f, st))
    out = []
    for v in by.values():
        C = np.array([s for _, s in sorted(v)], dtype=np.int32)
        out += [C[o::3] for o in range(3)]  # the 20 Hz streams the packets sample
    return out


def windows(seqs, stride=4):
    """-> inputs (N, L, 180) = [delta_t, abs_t] and targets (N, L, 90) = delta_{t+1}, all /32768."""
    X, Y = [], []
    for C in seqs:
        for b in range(0, len(C) - L - 2, stride):
            seg = C[b:b + L + 2]
            d = wrap16(seg[1:] - seg[:-1])  # (L+1, 90)
            X.append(np.concatenate([d[:L], seg[1:L + 1]], axis=1))
            Y.append(d[1:L + 1])
    return torch.tensor(np.array(X) / SC, dtype=torch.float32), torch.tensor(np.array(Y) / SC, dtype=torch.float32)


class GDN(nn.Module):
    """Gated DeltaNet-2 (arXiv 2605.22791), state held as S^T (dv x dk):
       S^T_t = (S^T_{t-1} D_t) - ((S^T_{t-1} D_t)(b_t*k_t) - w_t*v_t) k_t^T   with channel-wise decay D, erase b (dk), write w (dv)."""
    def __init__(self, H=2, dk=32, dv=32, hid=256):
        super().__init__()
        self.H, self.dk, self.dv = H, dk, dv
        self.q, self.k = nn.Linear(180, H * dk), nn.Linear(180, H * dk)
        self.v = nn.Linear(180, H * dv)
        self.a = nn.Linear(180, H * dk)   # decay per key channel
        self.b = nn.Linear(180, H * dk)   # erase gate per key channel
        self.w = nn.Linear(180, H * dv)   # write gate per value channel
        nn.init.constant_(self.a.bias, 2.0)
        self.mlp = nn.Sequential(nn.Linear(H * dv + 180, hid), nn.ReLU(), nn.Linear(hid, 90))

    def forward(self, x):  # (B, L, 180) -> (B, L, 90)
        B, Lx, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv
        q = F.normalize(self.q(x).view(B, Lx, H, dk), dim=-1, eps=1e-4)
        k = F.normalize(self.k(x).view(B, Lx, H, dk), dim=-1, eps=1e-4)
        v = self.v(x).view(B, Lx, H, dv)
        al = torch.sigmoid(self.a(x)).view(B, Lx, H, dk)
        be = torch.sigmoid(self.b(x)).view(B, Lx, H, dk)
        wr = torch.sigmoid(self.w(x)).view(B, Lx, H, dv)
        St = torch.zeros(B, H, dv, dk)
        outs = []
        for t in range(Lx):
            Sd = St * al[:, t, :, None, :]
            e = be[:, t] * k[:, t]
            r = torch.einsum("bhvk,bhk->bhv", Sd, e)
            St = Sd + (wr[:, t] * v[:, t] - r)[..., None] * k[:, t][:, :, None, :]
            outs.append(torch.einsum("bhvk,bhk->bhv", St, q[:, t]).reshape(B, -1))
        return self.mlp(torch.cat([torch.stack(outs, 1), x], -1))


def fit(model, X, Y, epochs, lr=2e-3, soft=None, alpha=0.5, log="", bs=256, warm=4):
    opt = torch.optim.Adam(model.parameters(), lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            p = model(X[idx])[:, warm:]
            y = Y[idx][:, warm:]
            loss = ((p - y) ** 2).mean() if soft is None else \
                alpha * ((p - soft[idx][:, warm:]) ** 2).mean() + (1 - alpha) * ((p - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sch.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  {log} epoch {ep+1}/{epochs} loss {tot/len(X):.3e}", flush=True)
    return model


# ---- integer student
SIG = np.rint(1.0 / (1.0 + np.exp(-((np.arange(16384) + 0.5) / 1024.0 - 8.0))) * (1 << FX)).astype(np.int64)


def sig_int(x):  # x in fixed point -> sigmoid in fixed point, by table
    return SIG[np.clip((x + (8 << FX)) >> 6, 0, 16383)]


def l2n_int(x):  # (..., d) fixed point -> unit vectors in fixed point, integer square root
    ss = (x * x).sum(-1)
    n = np.vectorize(math.isqrt, otypes=[np.int64])(ss)
    n = np.maximum(n, 1 << 4)[..., None]
    return (x << FX) // n


class IntGDN:
    def __init__(self, m):
        f = lambda t: np.rint(t.detach().numpy().astype(np.float64) * (1 << FX)).astype(np.int64)
        self.H, self.dk, self.dv = m.H, m.dk, m.dv
        self.q, self.k, self.v, self.a, self.b, self.w = [(f(l.weight), f(l.bias)) for l in (m.q, m.k, m.v, m.a, m.b, m.w)]
        self.m1, self.m2 = [(f(l.weight), f(l.bias)) for l in m.mlp if isinstance(l, nn.Linear)]

    @staticmethod
    def lin(wb, x):
        return ((x @ wb[0].T + (1 << (FX - 1))) >> FX) + wb[1]

    def step(self, St, x_codes):
        """St (n,H,dv,dk) fixed point; x_codes (n,180) int codes -> (new state, predicted delta codes (n,90))."""
        n, R = len(x_codes), 1 << (FX - 1)
        x = (x_codes.astype(np.int64) << FX) // 32768
        q = l2n_int(self.lin(self.q, x).reshape(n, self.H, self.dk))
        k = l2n_int(self.lin(self.k, x).reshape(n, self.H, self.dk))
        v = self.lin(self.v, x).reshape(n, self.H, self.dv)
        al = sig_int(self.lin(self.a, x)).reshape(n, self.H, self.dk)
        be = sig_int(self.lin(self.b, x)).reshape(n, self.H, self.dk)
        wr = sig_int(self.lin(self.w, x)).reshape(n, self.H, self.dv)
        Sd = (St * al[:, :, None, :] + R) >> FX
        e = (be * k + R) >> FX
        r = ((Sd * e[:, :, None, :]).sum(-1) + R) >> FX
        z = (wr * v + R) >> FX
        St = Sd + (((z - r)[..., None] * k[:, :, None, :] + R) >> FX)
        o = ((St * q[:, :, None, :]).sum(-1) + R) >> FX
        h = np.maximum(self.lin(self.m1, np.concatenate([o.reshape(n, -1), x], 1)), 0)
        out = self.lin(self.m2, h)
        return St, np.clip((out * 32768 + R) >> FX, -32767, 32767)


def stream(r):
    s1 = np.where(np.abs(r) <= 126, r, -128).astype(np.int8)
    s2 = np.where(np.abs(r) <= 126, 0, r).astype(np.int16)
    return s1.tobytes() + s2.tobytes()


def wire_cost(blobs):
    frames = [b"".join(t) for t in blobs]
    tot = 0
    for i in range(1, len(frames)):
        d = zstd.ZstdCompressionDict(frames[i - 1], dict_type=zstd.DICT_TYPE_RAWCONTENT)
        tot += len(zstd.ZstdCompressor(level=1, dict_data=d).compress(frames[i]))
    return tot / (len(frames) - 1)


if __name__ == "__main__":
    tr, va = tick_seqs("train"), tick_seqs("validation")
    testc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 90).astype(np.int32)
    Xt, Yt = windows(tr)
    Xv, Yv = windows(va, stride=8)
    print("windows train", tuple(Xt.shape), "validation", tuple(Xv.shape), flush=True)
    t0 = time.time()
    teacher = fit(GDN(H=4, dk=32, dv=32, hid=512), Xt, Yt, epochs=25, lr=1e-3, log="teacher gdn")
    teacher.eval()
    with torch.no_grad():
        soft = torch.cat([teacher(Xt[i:i + 512]) for i in range(0, len(Xt), 512)])
        pv_t = torch.cat([teacher(Xv[i:i + 512]) for i in range(0, len(Xv), 512)])
    rms = lambda p: float(((p[:, 4:] - Yv[:, 4:]) ** 2).mean().sqrt()) * SC
    print(f"teacher done {time.time()-t0:.0f}s", flush=True)
    torch.manual_seed(11)
    plain = fit(GDN(), Xt, Yt, epochs=40, log="student plain")
    torch.manual_seed(11)
    dist = fit(GDN(), Xt, Yt, epochs=40, soft=soft, alpha=0.5, log="student distilled")
    with torch.no_grad():
        pv_p, pv_d = plain(Xv), dist(Xv)
    last = torch.cat([torch.zeros_like(Yv[:, :1]), Yv[:, :-1]], 1)  # dead reckoning: the next delta equals the last
    print(f"validation RMS of the delta, codes: dead reckoning {rms(last):.0f}, student plain {rms(pv_p):.0f}, "
          f"student distilled {rms(pv_d):.0f}, teacher {rms(pv_t):.0f}", flush=True)

    ig = {"plain": IntGDN(plain), "distilled": IntGDN(dist)}
    for reset in (None, 32):
        tag = "continuous state" if reset is None else f"state reset every {reset} ticks"
        rows = {}
        for name in ("dead reckoning", "student plain (int)", "student distilled (int)"):
            blobs, absr = [], []
            St = None
            m = ig["plain"] if "plain" in name else ig["distilled"]
            st = np.zeros((8, m.H, m.dv, m.dk), dtype=np.int64) if name != "dead reckoning" else None
            pred_next = None
            for t in range(1, 400):
                C = testc
                if st is not None:
                    if reset and t % reset == 0:
                        st = np.zeros_like(st)
                    x = np.concatenate([wrap16(C[:, t] - C[:, t - 1]), C[:, t]], 1)
                    st, pnext = m.step(st, x)  # the prediction of delta_{t+1}, from exact data up to t
                if t + 1 >= T0 and t + 1 < 400:
                    row = []
                    for a in range(8):
                        pred = wrap16(C[a, t] - C[a, t - 1]) if st is None else pnext[a]
                        r = wrap16(C[a, t + 1] - C[a, t] - pred)
                        absr.append(np.abs(r))
                        row.append(stream(r))
                    blobs.append(row)
            c = wire_cost(blobs)
            e = np.concatenate(absr)
            rows[name] = c
            print(f"TEST [{tag}] {name:24s} {c:6.0f} B/tick = {c*8/240:5.1f} bits/joint | residual zero {np.mean(e==0)*100:4.1f}%, median {np.median(e):.0f}, escape {np.mean(e>126)*100:4.1f}%", flush=True)
    # the integer student against its own float model: the two ends only need to agree with each other, but check the quantisation
    with torch.no_grad():
        xf = torch.tensor(np.concatenate([wrap16(testc[:, 1:] - testc[:, :-1]), testc[:, 1:]], 2) / SC, dtype=torch.float32)
        pf = np.rint(dist(xf).numpy() * SC)
    m = ig["distilled"]
    st = np.zeros((8, m.H, m.dv, m.dk), dtype=np.int64)
    worst = 0
    for t in range(1, 60):
        st, pi = m.step(st, np.concatenate([wrap16(testc[:, t] - testc[:, t - 1]), testc[:, t]], 1))
        worst = max(worst, int(np.abs(pi - pf[:, t - 1]).max()))
    macs = 180 * 64 * 5 + 2 * 3 * 32 * 32 + (64 + 180) * 256 + 256 * 90
    print(f"student: ~{macs} MACs per avatar-tick, state {2*32*32} ints per avatar; integer vs float over 60 ticks: max {worst} codes")