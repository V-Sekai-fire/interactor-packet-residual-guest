# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Rung 3: an MLP that predicts the next rotation delta, as float, as 16-bit fixed point, and as a ternary net trained as
ternary from the first step (straight-through rounding). Every variant is measured in the same lossless wire format, and
every packet is decoded back and compared. Train on `train`, report the test avatars of frames.bin.
Not in this rung: the Hadamard fold of the weights, Gated DeltaNet-2 state, distillation. Each is the next rung."""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_gdn as g  # tick_seqs, wrap16, stream, wire_cost (its main is guarded)

torch.set_num_threads(4)
torch.manual_seed(7)
np.random.seed(7)
T = os.environ["TMPDIR"].rstrip("/")
SC, BLOCK, PIN = 32768.0, 128, 256


def pairs(seqs):
    X, Y = [], []
    for C in seqs:
        d = g.wrap16(C[1:] - C[:-1])  # d[i] = C[i+1] - C[i]
        X.append(np.concatenate([d[:-1], C[1:-1]], 1))  # [delta_t, abs_t]
        Y.append(d[1:])  # delta_{t+1}
    X, Y = np.concatenate(X), np.concatenate(Y)
    Xp = np.zeros((len(X), PIN), dtype=np.float32)
    Xp[:, :180] = X / SC
    return torch.tensor(Xp), torch.tensor(Y / SC, dtype=torch.float32)


def round_away(x):  # C roundf: halves away from zero
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


class TLinear(nn.Module):
    """Linear whose weights are ternary in the forward pass: absmean scale per block of 128 inputs, rounded to fp16 first."""
    def __init__(self, i, o, ternary):
        super().__init__()
        self.w = nn.Parameter(torch.randn(o, i) * (1.0 / i) ** 0.5)
        self.b = nn.Parameter(torch.zeros(o))
        self.ternary, self.i, self.o = ternary, i, o

    def quant(self):
        w = self.w.view(self.o, self.i // BLOCK, BLOCK)
        s = w.abs().mean(-1, keepdim=True).clamp_min(1e-8).half().float()
        q = torch.clamp(round_away(w / s), -1, 1)
        return q, s

    def forward(self, x):
        if not self.ternary:
            return x @ self.w.T + self.b
        q, s = self.quant()
        wq = (q * s).view(self.o, self.i)
        w = self.w + (wq - self.w).detach()  # straight-through
        return x @ w.T + self.b


class MLP(nn.Module):
    def __init__(self, ternary, h=256):
        super().__init__()
        self.l = nn.ModuleList([TLinear(PIN, h, ternary), TLinear(h, h, ternary), TLinear(h, 90, ternary)])

    def forward(self, x):
        x = torch.relu(self.l[0](x))
        x = torch.relu(self.l[1](x))
        return self.l[2](x)


def fit(m, X, Y, epochs, lr=2e-3, bs=512, log=""):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m.to(dev); X, Y = X.to(dev), Y.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=dev); tot = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            loss = ((m(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
        sch.step()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  {log} epoch {ep+1}/{epochs} loss {tot/len(X):.3e} [{dev}]", flush=True)
    return m.cpu()


FX = 16


def int_layers(m):
    """-> list of (w int8 (o,i), scale int32 (o,i/128) in 2^16 fixed point, bias int32), for the ternary net;
    for the float net, the weight itself in 2^16 fixed point (w int64 (o,i), bias int64)."""
    out = []
    for l in m.l:
        with torch.no_grad():
            if l.ternary:
                q, s = l.quant()
                out.append((q.view(l.o, l.i).numpy().astype(np.int8), np.rint(s[:, :, 0].numpy() * (1 << FX)).astype(np.int32),
                            np.rint(l.b.numpy() * (1 << FX)).astype(np.int64)))
            else:
                out.append((np.rint(l.w.numpy().astype(np.float64) * (1 << FX)).astype(np.int64), None,
                            np.rint(l.b.numpy().astype(np.float64) * (1 << FX)).astype(np.int64)))
    return out


def run_int(layers, x_fx):  # x_fx (n, PIN) int64 at 2^16; exact integer arithmetic
    a = x_fx.astype(np.int64)
    for k, (w, s, b) in enumerate(layers):
        if s is None:
            a = ((a @ w.T + (1 << (FX - 1))) >> FX) + b
        else:
            acc = np.zeros((len(a), w.shape[0]), dtype=np.int64)
            for bl in range(w.shape[1] // BLOCK):
                t = a[:, bl * BLOCK:(bl + 1) * BLOCK] @ w[:, bl * BLOCK:(bl + 1) * BLOCK].astype(np.int64).T
                acc += t * s[:, bl].astype(np.int64)[None]
            a = ((acc + (1 << (FX - 1))) >> FX) + b
        if k < len(layers) - 1:
            a = np.maximum(a, 0)
    return np.clip((a * 32768 + (1 << (FX - 1))) >> FX, -32767, 32767)


def measure(name, predict, testc):
    blobs, ab, ok = [], [], True
    for t in range(8, 399):
        row = []
        for a in range(8):
            pass
        blobs.append(row)
    # batch over the 8 avatars per tick: one call, as the codec would
    blobs, ab = [], []
    for t in range(8, 399):
        C = testc[:, t]  # (8,90) = frame t
        x = np.zeros((8, PIN), dtype=np.int64)
        x[:, :90] = g.wrap16(testc[:, t] - testc[:, t - 1]); x[:, 90:180] = testc[:, t]
        pred = predict(x)  # prediction of delta_{t+1}, from exact data up to t
        true = g.wrap16(testc[:, t + 1] - testc[:, t])
        r = g.wrap16(true - pred)
        back = g.wrap16(pred + r)  # the decoder, which adds the residual to the same prediction
        ok &= bool(np.array_equal(back, true))
        ab.append(np.abs(r)); blobs.append([g.stream(r[a]) for a in range(8)])
    c = g.wire_cost(blobs); e = np.concatenate(ab).ravel()
    print(f"TEST {name:34s} {c:6.0f} B/tick = {c*8/240:5.1f} bits/joint | residual zero {np.mean(e==0)*100:4.1f}%, median {np.median(e):.0f}, escape {np.mean(e>126)*100:4.1f}% | decode back exact: {ok}", flush=True)
    return c


if __name__ == "__main__":
    testc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 90).astype(np.int32)
    X, Y = pairs(g.tick_seqs("train")); Xv, Yv = pairs(g.tick_seqs("validation"))
    print("pairs train", tuple(X.shape), "validation", tuple(Xv.shape), flush=True)
    t0 = time.time()
    torch.manual_seed(11); fl = fit(MLP(False), X, Y, 40, log="float")
    torch.manual_seed(11); tn = fit(MLP(True), X, Y, 40, log="ternary QAT")
    rms = lambda m: float(((m(Xv) - Yv) ** 2).mean().sqrt()) * SC
    with torch.no_grad():
        print(f"validation RMS of the delta, codes: dead reckoning {float(((Xv[:, :90]-Yv)**2).mean().sqrt())*SC:.0f}, float {rms(fl):.0f}, ternary QAT {rms(tn):.0f}  ({time.time()-t0:.0f}s)", flush=True)
    lf, lt = int_layers(fl), int_layers(tn)
    def pred_float(x):
        with torch.no_grad():
            return np.clip(np.rint(fl(torch.tensor(x * 1.0 / SC, dtype=torch.float32)).numpy() * SC), -32767, 32767).astype(np.int64)
    fx = lambda x: x * 2  # codes/32768 at 2^16 fixed point is codes*2
    measure("dead reckoning (last delta)", lambda x: x[:, :90], testc)
    measure("float MLP (decoded in float)", pred_float, testc)
    measure("float MLP, 16-bit fixed point", lambda x: run_int(lf, fx(x)), testc)
    measure("ternary QAT, integer (Slang layer)", lambda x: run_int(lt, fx(x)), testc)
    # the ternary QAT net with its weights rounded after training instead (control: PTQ of the float net)
    pt = MLP(True); pt.load_state_dict(fl.state_dict())
    measure("float net rounded to ternary after", lambda x: run_int(int_layers(pt), fx(x)), testc)
    # the ternary net's integer output against its own float forward pass
    with torch.no_grad():
        xs = torch.tensor(np.concatenate([g.wrap16(testc[:, 1:399] - testc[:, :398]), testc[:, 1:399]], 2).reshape(-1, 180) / SC, dtype=torch.float32)
        xp = torch.zeros(len(xs), PIN); xp[:, :180] = xs
        pf = np.rint(tn(xp).numpy() * SC)
    pi = run_int(lt, (xp.numpy() * SC * 2).astype(np.int64))
    print(f"ternary net, integer vs float forward over {len(pf)} samples: max {int(np.abs(pi - pf).max())} codes, mean {np.abs(pi-pf).mean():.2f}")
    nz = [float(np.mean(w != 0)) for w, s, b in lt]
    print("ternary weights non-zero:", [f"{v*100:.0f}%" for v in nz], "| model size", sum(w.size for w, _, _ in lt) // 4 // 1, "bytes at 2 bits/weight (scales and biases extra)")
