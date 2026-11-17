# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Rung 3e: how long a history does the predictor need? MLP over the last K deltas and the pose, L1 loss, dead-reckoning skip.
Float and ternary QAT, measured in the same lossless wire format on the same ticks (17..398) for every K."""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_gdn as g
import rung3 as r0
import rung3b as r

torch.set_num_threads(4)
T = os.environ["TMPDIR"].rstrip("/")
SC, BLOCK = 32768.0, 128


def pad_to(n):
    return (n + BLOCK - 1) // BLOCK * BLOCK


def pairs(seqs, K, pin):
    n = sum(max(0, len(C) - 1 - K) for C in seqs)
    Xp = np.zeros((n, pin), dtype=np.float32); Y = np.zeros((n, 90), dtype=np.float32); k = 0
    for C in seqs:
        d = g.wrap16(C[1:] - C[:-1])  # d[i] = C[i+1] - C[i]
        for i in range(K - 1, len(d) - 1):
            Xp[k, :90 * K] = d[i - K + 1:i + 1][::-1].reshape(-1) / SC  # newest delta first, then the pose
            Xp[k, 90 * K:90 * K + 90] = C[i + 1] / SC
            Y[k] = d[i + 1] / SC; k += 1
    return torch.tensor(Xp[:k]), torch.tensor(Y[:k])


class Net(nn.Module):
    def __init__(self, pin, ternary, h=256):
        super().__init__()
        self.l = nn.ModuleList([r0.TLinear(pin, h, ternary), r0.TLinear(h, h, ternary), r0.TLinear(h, 90, ternary)])

    def forward(self, x0):
        x = torch.relu(self.l[0](x0)); x = torch.relu(self.l[1](x))
        return self.l[2](x) + x0[:, :90]  # the newest delta is the dead-reckoning prediction


def evaluate(name, K, pin, predict, testc, t0=101):
    blobs, ab, ok = [], [], True
    for t in range(t0, 399):
        x = np.zeros((8, pin), dtype=np.int64)
        for k in range(K):
            x[:, 90 * k:90 * k + 90] = g.wrap16(testc[:, t - k] - testc[:, t - k - 1])
        x[:, 90 * K:90 * K + 90] = testc[:, t]
        pred = predict(x)
        true = g.wrap16(testc[:, t + 1] - testc[:, t])
        res = g.wrap16(true - pred)
        ok &= bool(np.array_equal(g.wrap16(pred + res), true))
        ab.append(np.abs(res)); blobs.append([g.stream(res[a]) for a in range(8)])
    c = g.wire_cost(blobs); e = np.concatenate(ab).ravel()
    print(f"TEST K={K:<2} {name:34s} {c:6.0f} B/tick = {c*8/240:5.1f} bits/joint | zero {np.mean(e==0)*100:4.1f}%, median {np.median(e):.0f}, escape {np.mean(e>126)*100:4.1f}% | exact {ok}", flush=True)
    return c


if __name__ == "__main__":
    testc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 90).astype(np.int32)
    tr = g.tick_seqs("train")
    for K in [int(v) for v in sys.argv[1:]] or (8, 32, 64, 100):
        pin = pad_to(90 * K + 90)
        X, Y = pairs(tr, K, pin)
        print(f"K={K}: input {pin} wide, {len(X)} training windows", flush=True)
        evaluate("dead reckoning", K, pin, lambda x: x[:, :90], testc)
        for tern in (False, True):
            torch.manual_seed(11)
            m = r.fit(Net(pin, tern), X, Y, 40, l1=True, log=f"K={K} {'ternary' if tern else 'float'}")
            lt = r.int_layers(m)
            evaluate(f"{'ternary QAT' if tern else 'float, 16-bit fixed'}, L1 + skip", K, pin,
                     lambda x, lt=lt: np.clip(r.run_int(lt, x * 2) + x[:, :90], -32767, 32767), testc)
