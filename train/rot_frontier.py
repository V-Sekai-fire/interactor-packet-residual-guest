# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Frontier learning on the rotation representation only. The 8 held-out avatars, ticks 101..398, 30 joints, three codes each (swing_x, swing_z, twist).

A level is component x joint band (worst / middle / best 10 joints by dead-reckoning bits) x clip. A problem is one joint at one tick.
Success of a problem: its residual fits one byte (|r| <= 126); exact zero is counted too. A representation stays only when the success rate rises on
the worst-band levels without falling on the others. Every predictor extrapolates from the two previous ticks and goes through the same exact residual,
so none can change decodability, only size.

numpy matrix <-> code functions follow tools/pose_codes.py (checked against it below)."""
import math, os, sys
import numpy as np

SS = 32767.0 / math.pi
TS = 65536.0 / (2.0 * math.pi)
T = os.environ["TMPDIR"].rstrip("/")
CLIPS = ["walk_1", "jog_1", "run_1", "walk_2", "circles_1", "snake_1", "walk_turns_1", "crouching_1"]


def wrap16(a):
    return ((a + 32768) % 65536) - 32768


def rnd(v):  # halves away from zero
    return (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int64)


def swing_m(vx, vz):
    ang = np.hypot(vx, vz)
    safe = np.where(ang == 0, 1.0, ang)
    ax, az = vx / safe, vz / safe
    s, c = np.sin(ang), np.cos(ang)
    k = 1 - c
    z = np.zeros_like(ang)
    o = np.ones_like(ang)
    r = np.stack([c + k * ax * ax, -s * az, k * ax * az, s * az, c, -s * ax, k * ax * az, s * ax, c + k * az * az], -1)
    ident = np.stack([o, z, z, z, o, z, z, z, o], -1)
    return np.where((ang == 0)[..., None], ident, r).reshape(ang.shape + (3, 3))


def mats(c):
    sx, sz, tw = c[..., 0] / SS, c[..., 1] / SS, c[..., 2] / TS
    S = swing_m(sx, sz)
    cs, sn = np.cos(tw), np.sin(tw)
    z, o = np.zeros_like(tw), np.ones_like(tw)
    Tm = np.stack([cs, z, sn, z, o, z, -sn, z, cs], -1).reshape(tw.shape + (3, 3))
    return S @ Tm


def codes(R):
    ux, uy, uz = R[..., 0, 1], R[..., 1, 1], R[..., 2, 1]
    h = np.hypot(ux, uz)
    ang = np.arctan2(h, uy)
    hs = np.where(h > 1e-12, h, 1.0)
    vx = np.where(h > 1e-12, ang * uz / hs, np.where(uy < 0, math.pi, 0.0))
    vz = np.where(h > 1e-12, -ang * ux / hs, 0.0)
    S = swing_m(vx, vz)
    t00 = S[..., 0, 0] * R[..., 0, 0] + S[..., 1, 0] * R[..., 1, 0] + S[..., 2, 0] * R[..., 2, 0]
    t02 = S[..., 0, 0] * R[..., 0, 2] + S[..., 1, 0] * R[..., 1, 2] + S[..., 2, 0] * R[..., 2, 2]
    tw = wrap16(rnd(np.arctan2(t02, t00) * TS))
    cl = lambda v: np.clip(v, -32767, 32767)
    return np.stack([cl(rnd(vx * SS)), cl(rnd(vz * SS)), tw], -1)


def log_so3(R):  # rotation vector, atan2 form (no acos)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1) / 2
    s = np.linalg.norm(v, axis=-1)
    cth = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    ang = np.arctan2(s, cth)
    k = np.where(s > 1e-9, ang / np.where(s > 1e-9, s, 1.0), 1.0)
    return v * k[..., None]


def exp_so3(w):
    ang = np.linalg.norm(w, axis=-1)
    a = np.where(ang > 1e-12, ang, 1.0)
    n = w / a[..., None]
    s, c = np.sin(ang), np.cos(ang)
    k = 1 - c
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    R = np.stack([c + k * x * x, k * x * y - s * z, k * x * z + s * y,
                  k * x * y + s * z, c + k * y * y, k * y * z - s * x,
                  k * x * z - s * y, k * y * z + s * x, c + k * z * z], -1).reshape(ang.shape + (3, 3))
    return np.where((ang <= 1e-12)[..., None, None], np.eye(3), R)


def mT(a):
    return np.swapaxes(a, -1, -2)


# predictors: (c2, c1) -> predicted codes of the next tick. c arrays are (..., 3) ints.
def p_codes(c2, c1):
    return c1 + wrap16(c1 - c2)


def p_local(c2, c1, alpha=1.0):  # constant angular velocity in the joint's own frame
    R2, R1 = mats(c2), mats(c1)
    d = exp_so3(alpha * log_so3(mT(R2) @ R1))
    return codes(R1 @ d)


def p_world(c2, c1, alpha=1.0):  # constant angular velocity in the parent's frame
    R2, R1 = mats(c2), mats(c1)
    d = exp_so3(alpha * log_so3(R1 @ mT(R2)))
    return codes(d @ R1)


def p_hybrid_codes_twist(c2, c1):  # swing from the local-frame rotation, twist extrapolated as a code
    q = p_local(c2, c1)
    q[..., 2] = c1[..., 2] + wrap16(c1[..., 2] - c2[..., 2])
    return q


def p_swing_vec(c2, c1):  # swing by extrapolating the Y column u as a vector, twist as a code
    u2, u1 = mats(c2)[..., :, 1], mats(c1)[..., :, 1]
    u = 2 * u1 - u2
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    h = np.hypot(u[..., 0], u[..., 2])
    ang = np.arctan2(h, u[..., 1])
    hs = np.where(h > 1e-12, h, 1.0)
    vx = np.where(h > 1e-12, ang * u[..., 2] / hs, 0.0)
    vz = np.where(h > 1e-12, -ang * u[..., 0] / hs, 0.0)
    return np.stack([np.clip(rnd(vx * SS), -32767, 32767), np.clip(rnd(vz * SS), -32767, 32767), c1[..., 2] + wrap16(c1[..., 2] - c2[..., 2])], -1)


PRED = {
    "codes (dead reckoning)": p_codes,
    "local-frame, a=1.0": p_local,
    "local-frame, a=0.75": lambda a, b: p_local(a, b, 0.75),
    "local-frame, a=0.5": lambda a, b: p_local(a, b, 0.5),
    "parent-frame, a=1.0": p_world,
    "parent-frame, a=0.75": lambda a, b: p_world(a, b, 0.75),
    "local swing + code twist": p_hybrid_codes_twist,
    "Y-column vector + code twist": p_swing_vec,
}


def bits_of(a):  # proxy that follows the wire: 0 for an exact zero, log2|r|+2 up to 126, 16 for an escape
    return np.where(a == 0, 0.0, np.where(a > 126, 16.0, np.log2(np.maximum(a, 1)) + 2.0))


def check_against_scalar():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import pose_codes as pc
    rng = np.random.default_rng(3)
    r_ = 31000 * np.sqrt(rng.random(2000)); a_ = rng.random(2000) * 2 * np.pi  # swing magnitude kept under pi (a canonical swing)
    c = np.stack([rnd(r_ * np.cos(a_)), rnd(r_ * np.sin(a_)), rng.integers(-32768, 32767, 2000)], -1)
    m = mats(c)
    bad = 0
    for i in range(300):
        want = pc.codes_of_matrix(tuple(m[i].reshape(-1)))
        got = tuple(codes(m[i])[()] if False else codes(m[i:i + 1])[0])
        bad += max(abs(int(a) - int(b)) for a, b in zip(want, got)) > 1
    rt = np.abs(wrap16(codes(m) - c)).max()
    print(f"numpy vs tools/pose_codes.py: {bad}/300 differ by more than 1 code; code -> matrix -> code worst {rt}")
    assert bad == 0 and rt <= 2
    # control: a wrong sign must be caught
    bad_ctl = sum(max(abs(int(a) - int(b)) for a, b in zip(pc.codes_of_matrix(tuple(m[i].reshape(-1))), codes(mT(m[i:i + 1]))[0])) > 1 for i in range(100))
    print(f"control (transposed matrix): {bad_ctl}/100 differ")
    assert bad_ctl > 50


if __name__ == "__main__":
    check_against_scalar()
    tc = np.load(T + "/vae_testcodes.npy").reshape(8, 400, 30, 3).astype(np.int64)
    c2, c1, c0 = tc[:, 98:397], tc[:, 99:398], tc[:, 100:399]  # predict ticks 101..399 -> match where_bytes (101..398): 298 ticks
    c2, c1, c0 = tc[:, 99:397], tc[:, 100:398], tc[:, 101:399]
    base = np.abs(wrap16(c0 - PRED["codes (dead reckoning)"](c2, c1)))
    tot_b = bits_of(base)
    pj = tot_b.sum((0, 1, 3)); order = np.argsort(-pj)
    band = np.zeros(30, int); band[order[:10]] = 0; band[order[10:20]] = 1; band[order[20:]] = 2
    print(f"{c0.shape[1]} ticks; dead reckoning proxy {tot_b.sum() / (8 * c0.shape[1]):.0f} bits per avatar-tick")
    res = {}
    for name, f in PRED.items():
        res[name] = np.abs(wrap16(c0 - f(c2, c1)))
    comp = ("swing_x", "swing_z", "twist")
    print("\nall levels together (bits per avatar-tick | exact zero | fits a byte (<=126))")
    for name, a in res.items():
        print(f"  {name:30s} {bits_of(a).sum() / (8 * c0.shape[1]):7.0f} | {np.mean(a == 0) * 100:5.1f}% | {np.mean(a <= 126) * 100:5.1f}%")
    print("\nworst band (10 joints), fits a byte, by component:")
    print("  " + " " * 30 + "".join(f"{n:>10s}" for n in comp))
    for name, a in res.items():
        print(f"  {name:30s}" + "".join(f"{np.mean(a[:, :, band == 0, k] <= 126) * 100:9.1f}%" for k in range(3)))
    print("\nworst band, bits per avatar-tick by component:")
    for name, a in res.items():
        b = bits_of(a)
        print(f"  {name:30s}" + "".join(f"{b[:, :, band == 0, k].sum() / (8 * c0.shape[1]):10.0f}" for k in range(3)))
    # frontier: per level (component x band x clip) the representation with the fewest bits; keep a change only where success rises and no other level drops
    base_ok = {(k, b, i): np.mean(base[i][:, band == b, k] <= 126) for k in range(3) for b in range(3) for i in range(8)}
    print("\nfrontier, one component at a time (representation applied to that component only; the other two stay on codes):")
    for name, a in res.items():
        if name.startswith("codes"):
            continue
        row = []
        for k in range(3):
            ups = downs = 0; gain = 0.0
            for b in range(3):
                for i in range(8):
                    ok = np.mean(a[i][:, band == b, k] <= 126)
                    d = ok - base_ok[(k, b, i)]
                    ups += d > 0.005; downs += d < -0.005
            bitsd = (bits_of(base)[..., k].sum() - bits_of(a)[..., k].sum()) / (8 * c0.shape[1])
            row.append(f"{comp[k]} up {ups:2d} down {downs:2d} d_bits {bitsd:+6.1f}")
        print(f"  {name:30s} " + " | ".join(row))


def fit_alpha_per_joint(clips, grid=np.arange(0.0, 1.26, 0.125)):
    """Per joint and component, the damping a that minimises the proxy bits on the training clips (local-frame predictor, a applied to the rotation step)."""
    tot = np.zeros((len(grid), 30, 3))
    for c in clips:
        c2, c1, c0 = c[:-2], c[1:-1], c[2:]
        for gi, a in enumerate(grid):
            tot[gi] += bits_of(np.abs(wrap16(c0 - p_local(c2, c1, a)))).sum(0)
    return grid[np.argmin(tot, 0)], tot


def predict_with_alpha(c2, c1, alpha):  # alpha (30, 3): one per joint and component; the three predictions share a rotation step, each component takes its own damping
    out = np.empty_like(c1)
    for k in range(3):
        for a in np.unique(alpha[:, k]):
            m = alpha[:, k] == a
            out[..., m, k] = p_local(c2[..., m, :], c1[..., m, :], a)[..., k]
    return out
