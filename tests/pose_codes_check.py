# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Checks for tools/pose_codes.py (matrix -> swing_x, swing_z, twist, no quaternion).
  POSE_CLIP=path/to/clip.f32 python3 tests/pose_codes_check.py     (30 joints x 12 floats per frame, as the viewer's motion/mm)
Controls: swapped swing axes, a flipped twist sign, and a matrix that is not a rotation's twist must each be caught."""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import pose_codes as pc

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), name, detail)


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    return {"x": (1, 0, 0, 0, c, -s, 0, s, c), "y": (c, 0, s, 0, 1, 0, -s, 0, c), "z": (c, -s, 0, s, c, 0, 0, 0, 1)}[axis]


def angle(a, b):  # the turn between two matrices, atan2 of the antisymmetric part (acos of a trace is ill-conditioned near 0)
    A, B = np.array(a).reshape(3, 3), np.array(b).reshape(3, 3)
    r = A.T @ B
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2
    return math.atan2(np.linalg.norm(v), (np.trace(r) - 1) / 2)


def random_rotation(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); x, y, z, w = q
    return (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
            2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))


# the contract's worked examples (SWING_TWIST.md)
for label, m, want in [("identity", rot("x", 0), (0, 0, 0)), ("90 degrees about X", rot("x", math.pi / 2), (16383, 0, 0)),
                       ("90 degrees about Z", rot("z", math.pi / 2), (0, 16383, 0)), ("90 degrees about Y", rot("y", math.pi / 2), (0, 0, 16384))]:
    got = pc.codes_of_matrix(m)
    # A 90 degree swing is exactly 16383.5 codes. The contract's quaternion chain lands a hair under (16383); the direct route
    # lands on the tie and rounds away from zero (16384). Either is within one code; the same function runs at both ends.
    check(f"contract example, {label}", all(abs(g - w) <= 1 for g, w in zip(got, want)), f"got {got}, contract prints {want}")

rng = np.random.default_rng(1)
R = [random_rotation(rng) for _ in range(3000)]
worst = max(angle(m, pc.matrix_of_codes(*pc.codes_of_matrix(m))) for m in R)
check("3000 random rotations round trip", worst < 2.0e-4, f"worst {worst*1e6:.0f} microradians (one code is about 96)")


def swapped(m):
    sx, sz, tw = pc.codes_of_matrix(m)
    return pc.matrix_of_codes(sz, sx, tw)


def flipped(m):
    sx, sz, tw = pc.codes_of_matrix(m)
    return pc.matrix_of_codes(sx, sz, -tw)


for name, f in [("swing_x and swing_z swapped", swapped), ("twist sign flipped", flipped)]:
    bad = sum(angle(m, f(m)) > 2.0e-4 for m in R[:300])
    check(f"CONTROL {name} is caught", bad > 250, f"{bad} of 300 rotations fail")

# the swing's axis lies in the XZ plane, and S e_y = R e_y (the property the method rests on)
sx, sz, tw = pc.codes_of_matrix(R[0])
S = pc._swing_matrix(sx / pc.SWING_SCALE, sz / pc.SWING_SCALE)
check("swing takes +Y where the matrix takes it", max(abs(S[1] - R[0][1]), abs(S[4] - R[0][4]), abs(S[7] - R[0][7])) < 3e-4)

# the swing of pi, from the +Y axis flipped: the codes are still in range and decode to the same matrix
m = rot("x", math.pi)
check("swing of pi stays in range and round trips", angle(m, pc.matrix_of_codes(*pc.codes_of_matrix(m))) < 2.0e-4, f"codes {pc.codes_of_matrix(m)}")

path = os.environ.get("POSE_CLIP")
if path and os.path.exists(path):
    n = os.path.getsize(path) // (30 * 48)
    data = np.fromfile(path, dtype="<f4").reshape(n, 30, 12)
    worst, cnt = 0.0, 0
    for f in range(0, n, max(1, n // 80)):
        for j in range(30):
            m = tuple(float(v) for v in data[f, j, :9])
            worst = max(worst, angle(m, pc.matrix_of_codes(*pc.codes_of_matrix(m)))); cnt += 1
    check(f"{cnt} joint rotations of {os.path.basename(path)}", worst < 2.0e-4, f"worst {worst*1e6:.0f} microradians")
print("RESULT", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
