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

W, D, DT = int(os.environ.get("PAE_W", "32")), 8, 0.05  # 20 Hz ticks
SCALE = 30.0
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def six(codes):  # (T,30,3) int codes -> (T,180) float: the first two columns of each joint's matrix
    return r.mats(codes)[..., :, :2].reshape(len(codes), 30 * 6).astype(np.float32)


QAT = os.environ.get("QAT", "0") == "1"


class TLinear(nn.Linear):  # ternary weights in the forward pass, straight-through gradient to the float copy
    def forward(self, x):
        w = self.weight
        return nn.functional.linear(x, w + (qternary(w) - w).detach() if QAT else w, self.bias)


class TConv(nn.Conv1d):
    def forward(self, x):
        w = self.weight
        return self._conv_forward(x, w + (qternary(w) - w).detach() if QAT else w, self.bias)


class GDN2Enc(nn.Module):
    """Gated DeltaNet-2 (arXiv 2605.22791, clean-room from the equations) as the encoder: per tick, S^T_t = (S^T_{t-1} Dec_t) - ((S^T_{t-1} Dec_t)(b_t * k_t) - w_t * v_t) k_t^T with
    channel-wise decay, erase and write gates; o_t = S^T_t q_t; a per-tick linear reads the D latent values from o_t. (B, 180, W) -> (B, D, W)."""
    def __init__(self, H=int(os.environ.get("GH", "2")), dk=int(os.environ.get("GK", "32")), dv=int(os.environ.get("GV", "32"))):
        super().__init__()
        self.H, self.dk, self.dv = H, dk, dv
        self.q, self.k, self.v = TLinear(IND, H * dk), TLinear(IND, H * dk), TLinear(IND, H * dv)  # the gates a, b, w stay float
        self.a, self.b, self.w = nn.Linear(IND, H * dk), nn.Linear(IND, H * dk), nn.Linear(IND, H * dv)
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
EXTRA = int(os.environ.get("PAE_EXTRA", "0"))  # side-information channels (a token) given to the decoder, same for every tick of the window


class AttnEnc(nn.Module):
    """Causal attention over the window: every tick queries itself and all earlier ticks (keys and values are the window's own ticks), 4 heads, learned position embedding, residual + small feed-forward -> D values per tick."""
    def __init__(self, d=64):
        super().__init__()
        self.proj = nn.Linear(IND, d); self.pos = nn.Parameter(torch.randn(W, d) * 0.02)
        self.att = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 128), nn.ELU(), nn.Linear(128, D))
        self.register_buffer("mask", torch.triu(torch.ones(W, W, dtype=torch.bool), 1))

    def forward(self, x):  # (B, IND, W) -> (B, D, W)
        h = self.proj(x.transpose(1, 2)) + self.pos
        h = h + self.att(h, h, h, attn_mask=self.mask, need_weights=False)[0]
        return self.ff(h).transpose(1, 2)

TS = os.environ.get("VAE_TS", "0") == "1"  # teacher sees the next tick, student only the past; KL(teacher || student prior) is the bottleneck
VAE = float(os.environ.get("VAE_BETA", "-1"))  # >= 0: replace the sinusoid with Gaussian latents and a KL(q || N(0,1)) rate term of weight beta (the standard information bottleneck)


class PAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = GDN2Enc() if ENC == "gdn2" else AttnEnc() if ENC == "attn" else nn.Sequential(nn.Conv1d(IND, 96, 5, padding=2), nn.ELU(), nn.Conv1d(96, D, 5, padding=2))
        self.phase = nn.Linear(W, 2)
        self.lv = nn.Conv1d(IND, D, 5, padding=2)  # log-variance head, used only in the VAE mode
        self.kl = torch.zeros(())
        if TS:
            self.tenc = nn.Sequential(nn.Conv1d(180, 96, 5, padding=2), nn.ELU(), nn.Conv1d(96, 2 * D, 5, padding=2))  # teacher: 6D window including the target tick
            self.senc = nn.Sequential(nn.Conv1d(IND, 96, 5, padding=2), nn.ELU(), nn.Conv1d(96, 3 * D, 5, padding=2))  # student: features, prior mean, prior log-variance
            self.dec = nn.Sequential(TConv(2 * D, 96, 5, padding=2), nn.ELU(), TConv(96, 180, 5, padding=2))
        self.dec = nn.Sequential(TConv(D + EXTRA, 96, 5, padding=2), nn.ELU(), TConv(96, 180, 5, padding=2))
        k = torch.arange(1, W // 2 + 1).float(); t = torch.arange(W).float() * DT
        self.register_buffer("fk", k / (W * DT))
        self.register_buffer("C", torch.cos(2 * np.pi * self.fk[:, None] * t[None]))
        self.register_buffer("S", torch.sin(2 * np.pi * self.fk[:, None] * t[None]))
        self.register_buffer("tt", torch.arange(W + 1).float() * DT)
        if TS: self.dec = nn.Sequential(TConv(2 * D, 96, 5, padding=2), nn.ELU(), TConv(96, 180, 5, padding=2))
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

    def ts(self, x, fut=None):
        sx = self.senc(x); h, mup, lvp = sx[:, :D], sx[:, D:2 * D, -1], sx[:, 2 * D:, -1].clamp(-8, 4)
        if fut is not None:
            t = self.tenc(fut)[:, :, -1]; muq, lvq = t[:, :D], t[:, D:].clamp(-8, 4)
            z = muq + torch.randn_like(muq) * torch.exp(0.5 * lvq) if self.training else muq
            self.kl = 0.5 * (lvp - lvq + (lvq.exp() + (muq - mup) ** 2) / lvp.exp() - 1).sum(1).mean()  # KL(q || p), nats per avatar-tick
        else:
            z = mup
        hh = torch.cat([h, h[:, :, -1:]], 2)
        o = self.dec(torch.cat([hh, z[:, :, None].expand(-1, -1, W + 1)], 1))
        last = torch.sign(o[:, :, -1]) * torch.relu(o[:, :, -1].abs() - self.tau.abs() * SCALE)
        return torch.cat([o[:, :, :-1], last[:, :, None]], 2)

    def forward(self, x, fscale=None, fut=None, extra=None):
        if TS: return self.ts(x, fut)
        if VAE >= 0:
            mu = self.enc(x); lv = self.lv(x).clamp(-8, 4)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
            self.kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(1).mean()  # nats per tick of latent
            zz = torch.cat([z, z[:, :, -1:]], 2)  # the decoder predicts W + 1 ticks from W latent ticks
            if EXTRA: zz = torch.cat([zz, (extra if extra is not None else torch.zeros(zz.shape[0], EXTRA, device=zz.device))[:, :, None].expand(-1, -1, W + 1)], 1)
            o = self.dec(zz)
            last = torch.sign(o[:, :, -1]) * torch.relu(o[:, :, -1].abs() - self.tau.abs() * SCALE)
            return torch.cat([o[:, :, :-1], last[:, :, None]], 2)
        f, A, B, s = self.params(x)
        if fscale is not None: f = f * fscale
        curve = A[..., None] * torch.sin(2 * np.pi * (f[..., None] * self.tt - s[..., None])) + B[..., None]
        o = self.dec(curve)  # (B, 180, W+1): offsets from the last observed tick, scaled; the last entry is a correction to dead reckoning
        last = torch.sign(o[:, :, -1]) * torch.relu(o[:, :, -1].abs() - self.tau.abs() * SCALE)
        return torch.cat([o[:, :, :-1], last[:, :, None]], 2)


IN = os.environ.get("PAE_IN", "six")  # six | codes | ts
IND = 210 if IN == "ts" else 180  # six: 6D matrix columns (smooth); codes: [wrapped code delta, absolute code] / 32768, the form the earlier GDN-2 students saw


def feat(codes):
    c = codes.reshape(len(codes), 90).astype(np.int64)
    d = np.zeros_like(c); d[1:] = r.wrap16(c[1:] - c[:-1])
    return (np.concatenate([d, c], 1) / 32768.0).astype(np.float32)


def feat_ts(codes):
    c = codes.reshape(len(codes), 30, 3).astype(np.int64)
    d = np.zeros_like(c); d[1:] = r.wrap16(c[1:] - c[:-1])
    tw = c[:, :, 2] / (65536.0 / (2 * np.pi))
    return np.concatenate([d.reshape(len(c), 90) / 32768.0, c[:, :, :2].reshape(len(c), 60) / 32768.0, np.cos(tw), np.sin(tw)], 1).astype(np.float32)


def load_train():
    names = set(r.CLIPS); seqs = []
    for f in sorted(glob.glob(r.T + "/more/*.npy")):
        if os.path.basename(f)[:-4] in names: continue
        C = np.load(f).astype(np.int64).reshape(-1, 30, 3); C[:, :, :2] *= -1
        for off in range(3):
            q = C[off::3]
            if len(q) > W + 5: seqs.append((six(q), feat_ts(q) if IN == "ts" else feat(q)))
    return seqs


def rel_target(win):
    """win (B,180,W+1) -> targets: ticks 0..W-1 as offsets from tick W-1; the last entry is the second difference (the correction dead reckoning needs), all scaled."""
    ref = win[:, :, W - 1:W]
    rel = (win[:, :, :W] - ref) * SCALE
    corr = (win[:, :, W] - 2 * win[:, :, W - 1] + win[:, :, W - 2]) * SCALE
    return torch.cat([rel, corr[:, :, None]], 2)


def qsym(w, bits):  # symmetric per-output-row quantisation to a signed integer grid
    flat = w.reshape(w.shape[0], -1); qmax = 2 ** (bits - 1) - 1
    sc = flat.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
    return ((flat / sc).round().clamp(-qmax, qmax) * sc).reshape(w.shape)


def qternary(w, block=128):  # ternary {-1,0,+1} with an absmean scale per block of 128 along each row (Bonsai 2 style)
    flat = w.reshape(w.shape[0], -1); n = flat.shape[1]; pad = (-n) % block
    f = torch.nn.functional.pad(flat, (0, pad)).reshape(w.shape[0], -1, block)
    sc = f.abs().mean(-1, keepdim=True).clamp_min(1e-12)
    return (torch.clamp((f / sc).round(), -1, 1) * sc).reshape(w.shape[0], -1)[:, :n].reshape(w.shape)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    seqs = load_train(); lens = [len(s[0]) for s in seqs]
    X = torch.tensor(np.concatenate([q[0] for q in seqs])).to(DEV); XF = torch.tensor(np.concatenate([q[1] for q in seqs])).to(DEV)
    starts = np.concatenate([np.arange(o, o + L - W) for o, L in zip(np.cumsum([0] + lens[:-1]), lens)])
    print(f"encoder {ENC}; {len(seqs)} sequences, {sum(lens)} ticks, {len(starts)} windows, device {DEV}", flush=True)
    m = PAE().to(DEV); m.train(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    idx_t = torch.tensor(starts).to(DEV); ar = torch.arange(W + 1).to(DEV)
    wt = torch.ones(W + 1).to(DEV); wt[-1] = float(os.environ.get("LASTW", "1000"))
    t0 = time.time()
    for it in range(steps):
        b = idx_t[torch.randint(len(idx_t), (256,), device=DEV)]
        win = X[b[:, None] + ar[None]].transpose(1, 2)                  # (B, 180, W+1)
        inp = (X if IN == "six" else XF)[b[:, None] + ar[None]].transpose(1, 2)
        out = m(inp[:, :, :W], None, win if TS else None)
        tgt = rel_target(win)
        loss = ((out - tgt).abs().mean(1) * wt).mean() + (VAE * m.kl if VAE >= 0 else 0)
        if TS: loss = loss + ((m(inp[:, :, :W]) - tgt).abs().mean(1) * wt).mean()  # the student alone, with its own prior, must also predict
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0 or it == steps - 1: print(f"step {it:5d} loss {loss.item():.5f}  {time.time() - t0:.0f}s", flush=True)
    m.eval()
    evaluate(m, steps)


def evaluate(m, steps, quant=True):
    tc = np.load(r.T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64)
    START = 101
    def predict(fscale=None, oracle=False):
        P = np.zeros((8, 400, 90), np.int64)
        for i in range(8):
            xs = torch.tensor(six(tc[i])).to(DEV); xi = torch.tensor({"six": six, "codes": feat, "ts": feat_ts}[IN](tc[i])).to(DEV)
            wins = torch.stack([xs[t - W:t].T for t in range(START, 400)])  # (T', 180, W) 6D, for dead reckoning
            winp = torch.stack([xi[t - W:t].T for t in range(START, 400)])  # what the encoder is given
            fut = torch.stack([xs[t - W:t + 1].T for t in range(START, 400)]) if oracle else None
            with torch.no_grad(): corr = m(winp, None if fscale is None else fscale, fut)[:, :, -1] / SCALE
            out = (2 * wins[:, :, -1] - wins[:, :, -2] + corr).cpu().numpy()  # dead reckoning in 6D plus the correction
            R = oe.six_to_rot(out.reshape(-1, 30, 3, 2).astype(np.float64))
            P[i, START:] = r.codes(R).reshape(-1, 90)
        return P
    C = tc.reshape(8, 400, 90)
    if os.environ.get("PAE_DUMP") and quant:
        np.save(os.environ["PAE_DUMP"], predict()); print("dumped predicted codes (8 avatars, 400 ticks, 90 codes) to", os.environ["PAE_DUMP"])
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
    if TS:
        P3 = predict(oracle=True); res["oracle: the teacher's latent (needs the future)"] = r.wrap16(C[:, START:] - P3[:, START:])
    rng = np.random.default_rng(5); fs = torch.tensor(rng.uniform(0.5, 2.0, (1, D)).astype(np.float32)).to(DEV)
    P2 = predict(fs); res["control, frequencies scrambled"] = r.wrap16(C[:, START:] - P2[:, START:])
    # control: predict "no change" (copy the last tick) for scale
    res["copy the last tick"] = np.stack([r.wrap16(C[i, START:] - C[i, START - 1:-1]) for i in range(8)])
    A, B = [0, 1, 2, 3], [4, 5, 6, 7]
    c1, c2 = cp.costs(res, A, B), cp.costs(res, B, A)
    print("\nbits per avatar-tick (size class per joint-component, tables fitted on four avatars, scored on the other four):")
    for n in res: print(f"  {n:34s} {(c1[n] + c2[n]) / 2:6.0f}   exact zero {np.mean(res[n] == 0) * 100:4.1f}%  escape {np.mean(np.abs(res[n]) > 126) * 100:4.1f}%")
    if not quant: return (c1["learned phase manifold"] + c2["learned phase manifold"]) / 2
    f, Aa, Bb, s = m.params(torch.tensor(({"six": six, "codes": feat, "ts": feat_ts}[IN](tc[0]))[START - W:START].T[None]).to(DEV))
    if quant:
        import copy
        base_bits = (c1["learned phase manifold"] + c2["learned phase manifold"]) / 2
        print("\npost-training weight quantisation of the trained model (no retraining), bits per avatar-tick; dead reckoning 629:")
        for name, fn in (("float", None), ("16-bit", lambda w: qsym(w, 16)), ("8-bit", lambda w: qsym(w, 8)), ("4-bit", lambda w: qsym(w, 4)), ("ternary, blocks of 128", qternary)):
            mq = copy.deepcopy(m)
            if fn is not None:
                for n_, p_ in mq.named_parameters():
                    if p_.dim() >= 2 and "tau" not in n_: p_.data = fn(p_.data)
            res_q = evaluate(mq, steps, quant=False)
            print(f"  {name:24s} {res_q:6.0f}")
    if VAE >= 0: print(f"latent KL at the end of training: {m.kl.item():.2f} nats per latent tick (beta {VAE})"); return
    print("example latent frequencies (Hz) for walk_1:", np.round(f[0].detach().cpu().numpy(), 2))


if __name__ == "__main__":
    main()
