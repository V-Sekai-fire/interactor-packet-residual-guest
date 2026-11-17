# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT

"""The contract's Makima dead-reckoning kernel (EntityPacket/Predict.lean, entity_predict.h), in Python, against its own
golden rows. Every division truncates toward zero except the last, which rounds half away from zero."""
import csv, sys

PQ, RAW_MAX = 65536, 32767


def tdiv(a, b):  # C division: truncate toward zero
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def tang(p0, p1, p2, p3):
    w1 = 2 * abs(p3 - p2) + abs(p3 + p2)
    w2 = 2 * abs(p1 - p0) + abs(p1 + p0)
    s = w1 + w2
    return tdiv(p1 + p2, 2) if s == 0 else tdiv(w1 * p1 + w2 * p2, s)


def rdiv(n, d):
    return tdiv(2 * n + d, 2 * d) if n >= 0 else -tdiv(-2 * n + d, 2 * d)


def predict_disp_um(vpp, vp, vf, vt, lag, vmax):
    if lag <= 0:
        return 0
    l = lag
    s1 = tdiv(l * (l + 1), 2)
    s2 = tdiv(l * (l + 1) * (2 * l + 1), 6)
    s3 = s1 * s1
    ca, cb, cc, cd = 2 * s3 - 3 * l * s2 + l ** 4, s3 - 2 * l * s2 + l * l * s1, -2 * s3 + 3 * l * s2, s3 - l * s2
    a_from, a_pre, a_pp = (vt - vf) * PQ, l * (vf - vp) * PQ, l * (vp - vpp) * PQ
    a_post = 2 * a_from - a_pre
    a_post2 = 2 * a_post - a_from
    tf = tang(a_pp, a_pre, a_from, a_post)
    tt = tang(a_pre, a_from, a_post, a_post2)
    n = ca * vf * PQ + cb * tf + cc * vt * PQ + cd * tt
    d = rdiv(n * vmax, RAW_MAX * PQ * l ** 3)
    cap = vmax * l
    return max(-cap, min(cap, d))


def main(path, plant=False):
    rows = list(csv.DictReader(open(path)))
    bad = 0
    for r in rows:
        got = predict_disp_um(int(r["vpp"]), int(r["vp"]), int(r["vf"]), int(r["vt"]), int(r["lag"]), int(r["vmax_um_per_tick"]))
        want = int(r["disp_um"]) + (1 if plant else 0)
        bad += got != want
    return len(rows), bad


if __name__ == "__main__":
    n, bad = main(sys.argv[1])
    print(f"GOLDEN {n} rows, {bad} differ")
    n, bad_p = main(sys.argv[1], plant=True)
    print(f"CONTROL planted +1um: {bad_p} of {n} differ (want all)")
    sys.exit(0 if bad == 0 and bad_p == n else 1)
