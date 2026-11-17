# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""A rotation matrix to the packet's swing_x, swing_z, twist (contract-entity-packet SWING_TWIST.md), with no quaternion.

R = S T with the twist T a turn about +Y and the swing S a turn about an axis in the XZ plane, so R e_y = S e_y: the swing is
the shortest turn that takes +Y to the matrix's Y column u. Its rotation vector is angle * (u_z, 0, -u_x) / |(u_x, u_z)| with
angle = atan2(|(u_x, u_z)|, u_y), in [0, pi]. The twist is the turn left over: T = S^T R, angle = atan2(T[0][2], T[0][0]).
Matrices are row-major, nine numbers; column j is the image of axis j (Godot's Basis.x, y, z). Plain arithmetic, no quaternion, no acos.
At a swing of exactly pi the axis and the twist cannot be told apart: the axis is taken as +X (the contract's canonical sign)."""
import math

SWING_SCALE = 32767.0 / math.pi
TWIST_SCALE = 65536.0 / (2.0 * math.pi)


def _round(v):  # halves away from zero, as C roundf
    return int(math.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)


def _swing_matrix(vx, vz):  # Rodrigues for an axis in the XZ plane: angle = |v|
    ang = math.hypot(vx, vz)
    if ang == 0.0:
        return (1, 0, 0, 0, 1, 0, 0, 0, 1)
    ax, az = vx / ang, vz / ang
    s, c = math.sin(ang), math.cos(ang)
    k = 1.0 - c
    return (c + k * ax * ax, -s * az, k * ax * az,
            s * az, c, -s * ax,
            k * ax * az, s * ax, c + k * az * az)


def codes_of_matrix(m):
    """-> (swing_x, swing_z, twist) as the packet holds them."""
    ux, uy, uz = m[1], m[4], m[7]
    h = math.hypot(ux, uz)
    ang = math.atan2(h, uy)
    if h > 1e-12:
        vx, vz = ang * uz / h, -ang * ux / h
    elif uy < 0:  # swing of pi: any axis in the XZ plane; +X is the canonical one
        vx, vz = math.pi, 0.0
    else:
        vx, vz = 0.0, 0.0
    s = _swing_matrix(vx, vz)
    # T = S^T R: only its first row's x and z entries are needed
    t00 = s[0] * m[0] + s[3] * m[3] + s[6] * m[6]
    t02 = s[0] * m[2] + s[3] * m[5] + s[6] * m[8]
    twist = _round(math.atan2(t02, t00) * TWIST_SCALE)
    twist = ((twist + 32768) % 65536) - 32768  # circular: pi and -pi share a code
    clamp = lambda v: max(-32767, min(32767, v))
    return clamp(_round(vx * SWING_SCALE)), clamp(_round(vz * SWING_SCALE)), twist


def matrix_of_codes(sx, sz, tw):
    """The inverse, for tests: S T as nine numbers."""
    s = _swing_matrix(sx / SWING_SCALE, sz / SWING_SCALE)
    t = tw / TWIST_SCALE
    c, n = math.cos(t), math.sin(t)
    T = (c, 0, n, 0, 1, 0, -n, 0, c)
    return tuple(sum(s[3 * i + k] * T[3 * k + j] for k in range(3)) for i in range(3) for j in range(3))
