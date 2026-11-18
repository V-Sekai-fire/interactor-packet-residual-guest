# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""The whole 100 bytes, modelled by a Gated DeltaNet-2 with a VAE bottleneck, trained on the code length itself.
Per avatar-tick the coded wire is 30 joints x 100 bytes = 3000 bytes. Input at tick t: for every byte, (byte != 0) and byte / 255, plus the 6D rotation of the 30 joints (6180 values).
The Gated DeltaNet-2 layer (clean-room from arXiv 2605.22791, as in phase_manifold.py) carries a per-avatar state over the ticks; its output goes through a Gaussian bottleneck z (D values,
KL to N(0,1), weight VB); z picks the mixing weights of K static tables, one gate per joint x byte position: p(byte) = sum_k w_k(z)[ctx] * table_k[ctx][byte], with a context = joint x position
(3000), so with K = 1 and no input it is the static coder. Loss = ideal bits of tick t+1's bytes (cross entropy) + VB * KL. Train: the 26 library avatars (13 clips x 2 offsets, none a test
clip) in windows of L ticks; test: the 8 held-out avatars run through their whole clip, scored on ticks 101 and later. Float model, ideal code length: a measurement, not the shared integer decoder.
Controls: the static tables alone; the model fed another avatar's input; the model fed its own input shuffled in time. Needs PACKET_MODULE_DIR and the caches of more_training.py / dict_wire.py."""
import os, pickle, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import packet_residual_lossless as P
import rot_frontier as r

T = os.environ["TMPDIR"].rstrip("/")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
L, K, D, VB = int(os.environ.get("WL", "100")), int(os.environ.get("WK", "4")), 16, float(os.environ.get("VB", "0.01"))
H, DK, DV = int(os.environ.get("GH", "2")), int(os.environ.get("GK", "32")), int(os.environ.get("GV", "32"))
STEPS, BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 2000, 12
torch.manual_seed(0); np.random.seed(0)


def six(codes):  # (T,30,3) codes -> (T,180)
    return r.mats(codes)[..., :, :2].reshape(len(codes), 180).astype(np.float32)


def train_codes():
    cache = T + "/train_codes.npy"
    if os.path.exists(cache): return np.load(cache)
    data = open(T + "/frames_train.bin", "rb").read(); out = np.zeros((26, 400, 30, 3), np.int64)
    for i in range(len(data) // 100):
        t, rest = divmod(i, 780); a, j = divmod(rest, 30); v = P.vals(data[i * 100:(i + 1) * 100]); out[a, t, j] = (v["swing_x"], v["swing_z"], v["twist"])
    np.save(cache, out); return out


Wt = np.frombuffer(b"".join(pickle.load(open(T + "/coded_wire_train.pkl", "rb"))), np.uint8).reshape(400, 26, 3000).transpose(1, 0, 2).copy()  # our own caches
Wv = np.frombuffer(b"".join(pickle.load(open(T + "/coded_wire.pkl", "rb"))), np.uint8).reshape(400, 8, 3000).transpose(1, 0, 2).copy()
Rt = np.stack([six(c) for c in train_codes()]); Rv = np.stack([six(c) for c in np.load(T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64)])
Bt, Bv = torch.tensor(Wt).to(DEV), torch.tensor(Wv).to(DEV); Rt, Rv = torch.tensor(Rt).to(DEV), torch.tensor(Rv).to(DEV)


def feats(B, Rt_):  # (n, ticks, 3000) uint8, (n, ticks, 180) -> (n, ticks, 6180)
    return torch.cat([(B != 0).float(), B.float() / 255.0, Rt_], -1)


cnt = np.full((3000, 256), 0.05)
for a in range(26):
    np.add.at(cnt, (np.tile(np.arange(3000), 399), Wt[a, 1:].reshape(-1)), 1)
BASE = torch.tensor(np.log(cnt / cnt.sum(1, keepdims=True)), dtype=torch.float32).to(DEV)  # the static tables, log-probabilities


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6180, H * (4 * DK + 2 * DV))
        nn.init.constant_(self.proj.bias, 0.0); self.proj.bias.data[2 * H * DK:3 * H * DK] = 2.0  # decay gate a starts near 1
        self.mu, self.lv = nn.Linear(H * DV, D), nn.Linear(H * DV, D)
        self.gate = nn.Linear(D, 3000 * K); nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.tab = nn.Parameter(BASE[None].repeat(K, 1, 1) + 0.3 * torch.randn(K, 3000, 256, device=DEV) * (torch.arange(K, device=DEV) > 0).float()[:, None, None])

    def gdn(self, x):  # (B, L, 6180) -> (B, L, H*DV)
        Bn, Ln, _ = x.shape; p = self.proj(x); q, k, a, b, v, w = torch.split(p, [H * DK] * 4 + [H * DV] * 2, -1)
        q, k = [nn.functional.normalize(u.view(Bn, Ln, H, DK), dim=-1, eps=1e-4) for u in (q, k)]
        al, be, wr = torch.sigmoid(a).view(Bn, Ln, H, DK), torch.sigmoid(b).view(Bn, Ln, H, DK), torch.sigmoid(w).view(Bn, Ln, H, DV); v = v.view(Bn, Ln, H, DV)
        St = torch.zeros(Bn, H, DV, DK, device=x.device); outs = []
        for t in range(Ln):
            Sd = St * al[:, t, :, None, :]; rr = torch.einsum("bhvk,bhk->bhv", Sd, be[:, t] * k[:, t])
            St = Sd + (wr[:, t] * v[:, t] - rr)[..., None] * k[:, t][:, :, None, :]
            outs.append(torch.einsum("bhvk,bhk->bhv", St, q[:, t]).reshape(Bn, -1))
        return torch.stack(outs, 1)

    def forward(self, x, y):  # y (B, L, 3000) the next tick's bytes -> bits per tick (B, L), KL bits per tick (B, L)
        o = self.gdn(x); mu, lv = self.mu(o), self.lv(o).clamp(-8, 4)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1)
        g = torch.log_softmax(self.gate(z).view(*z.shape[:2], 3000, K), -1)
        tab = torch.log_softmax(self.tab, -1).permute(1, 0, 2)  # (3000, K, 256)
        idx = torch.arange(3000, device=x.device)
        lp = tab[idx[None, None, :], :, y.long()]  # (B, L, 3000, K)
        return -torch.logsumexp(g + lp, -1).sum(-1) / np.log(2), kl / np.log(2)


m = Model().to(DEV)
opt = torch.optim.Adam([{"params": [m.tab], "lr": 5e-3}, {"params": [p for n, p in m.named_parameters() if n != "tab"], "lr": 1e-3}])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
t0 = time.time()
for it in range(STEPS):
    a = torch.randint(26, (BATCH,)); s = torch.randint(1, 399 - L, (BATCH,)); tt = s[:, None] + torch.arange(L)[None]
    x = feats(Bt[a[:, None], tt], Rt[a[:, None], tt]); y = Bt[a[:, None], tt + 1]
    m.train(); bits, kl = m(x, y); loss = bits.mean() / 8 + VB * kl.mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 100 == 0 or it == STEPS - 1: print(f"step {it:5d} bytes per avatar-tick {bits.mean().item() / 8:6.1f}  KL {kl.mean().item():5.1f} bits  {time.time() - t0:.0f}s", flush=True)

m.eval()
with torch.no_grad():
    x = feats(Bv[:, 1:399], Rv[:, 1:399]); y = Bv[:, 2:400]; sl = slice(100, None)  # ticks 101 and later
    bits, kl = m(x, y); mod = bits[:, sl].mean().item() / 8
    other = m(torch.roll(x, 1, 0), y)[0][:, sl].mean().item() / 8
    shuf = m(x[:, torch.randperm(398)], y)[0][:, sl].mean().item() / 8
    static = -(BASE[torch.arange(3000, device=DEV)[None, None, :], y.long()]).sum(-1)[:, sl].mean().item() / np.log(2) / 8
print(f"\nideal bytes per avatar-tick on the 8 held-out avatars, ticks 101-399 (x4 = bytes per tick of four avatars):")
for name, v in (("static tables, joint x byte position (26 library avatars)", static), ("GDN-2 + VAE bottleneck, whole 100 bytes", mod), ("control: the model fed another avatar's input", other), ("control: the model fed its own ticks shuffled", shuf)):
    print(f"  {name:58s} {v:6.1f}  (x4 = {v * 4:5.0f})  {(v - static) / static * 100:+5.1f}%")
print(f"  bottleneck KL on the test avatars: {kl[:, sl].mean().item():.1f} bits per avatar-tick (side information if the latent were sent)")
