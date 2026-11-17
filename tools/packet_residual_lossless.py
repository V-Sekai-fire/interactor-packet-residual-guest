# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT

# Spike for issue #5: the current 100-byte entity packet, coded LOSSLESSLY as predicted + residual fix.
#   python packet_residual_lossless.py FRAMES.bin [ENTITIES=240] [TICKS=400]
# FRAMES.bin is ticks x ENTITIES x 100 B (mm_to_packets.py). Every byte of every packet comes back exact: gid,
# class_owner, sub_index, hlc, position, velocity, acceleration, rotation, payload and the reserved tail.
#
# A coded packet is still exactly 100 bytes, whole or malformed, in one of two forms told apart by byte 51 (the class
# byte of class_owner, little endian): a listed class (0..7, 128) is the old packet verbatim; 0xFF marks a residual
# packet. No source packet can carry 0xFF (the codec drops an unlisted class), so nothing is ambiguous and nothing is
# refused: a packet whose residuals do not fit takes the verbatim form.
#
# Residual packet:  0..15   stage 1, one int8 per scalar (0 = the prediction was exact, -128 = escape)
#                   16..50, 90..99   stage 2, the escaped residuals in field order (u32/i32 fields 4 B, i16 fields 2 B)
#                   51      marker 0xFF
#                   52..87  payload XOR the previous payload      88..89  reserved XOR the previous reserved
# Everything not written is zero, so zstd and run-length coding on the wire make it nearly free.
#
# The predictor is the part to swap: `predict(p1, p2, cur)` gets the previous two exact packets of the same slot and
# the fields already decoded of this one. Here it is dead reckoning plus mm_to_packets' own velocity and acceleration
# rule; the next stage is the guest ELF's physics and motion-matching picker, whose output is bit exact on both ends.
import os
import struct
import sys

import zstandard as zstd

# The packet module is emitted from contract-entity-packet (Lean) and is not copied here: point PACKET_MODULE_DIR at
# the directory that holds xr_grid_entity_packet.py (mixer-broadcaster keeps one at clients/python/tools).
sys.path.insert(0, os.environ.get("PACKET_MODULE_DIR", os.path.dirname(os.path.abspath(__file__))))
try:
    from xr_grid_entity_packet import SIZE, decode, encode  # noqa: E402
except ImportError:
    sys.exit("set PACKET_MODULE_DIR to the directory holding xr_grid_entity_packet.py (emitted from contract-entity-packet)")

V_MAX, A_MAX = 500000, 1000000
MARK, MARK_OFF = 0xFF, 51
ESC = -128
ESC_AT = list(range(16, 51)) + list(range(90, 100))
# scalar fields, in wire order: name, kind. kinds: u32 (modular), u64 (modular, escape 4 B signed), i64 (escape 4 B), i16
FIELDS = (
    ["gid", "class_owner", "sub_index", "hlc"]
    + [f"pos{k}" for k in "xyz"] + [f"vel{k}" for k in "xyz"] + [f"acc{k}" for k in "xyz"]
    + ["swing_x", "swing_z", "twist"]
)
WIDTH = {f: 2 if f.startswith(("vel", "acc", "swing", "twist")) else 4 for f in FIELDS}
MOD = {f: (1 << 16) if WIDTH[f] == 2 and f != "hlc" else (1 << 64) for f in FIELDS}
for f in ("gid", "class_owner", "sub_index"):
    MOD[f] = 1 << 32
MOD["hlc"] = 1 << 64
SIGNED = {f: f.startswith(("pos", "vel", "acc", "swing", "twist")) for f in FIELDS}


def vals(raw):
    p = decode(raw)
    return {
        "gid": p.gid, "class_owner": p.class_owner, "sub_index": p.sub_index, "hlc": p.hlc,
        "posx": p.pos_um_x, "posy": p.pos_um_y, "posz": p.pos_um_z,
        "velx": p.vel_x, "vely": p.vel_y, "velz": p.vel_z, "accx": p.accel_x, "accy": p.accel_y, "accz": p.accel_z,
        "swing_x": p.swing_x, "swing_z": p.swing_z, "twist": p.twist,
    }


def wrap(x, mod):
    x %= mod
    return x - mod if x >= mod // 2 else x


def to_range(v, f):  # store a value the way the field holds it (signed fields signed, the rest unsigned)
    return wrap(v, MOD[f]) if SIGNED[f] else v % MOD[f]


def clamp(x):
    return max(-32767, min(32767, x))


def predict(p1, p2, cur):
    """Prediction of every scalar of this packet. p1, p2 = previous two exact packets of the slot (p2 may be None);
    cur = the fields already decoded of this packet (position comes before velocity before acceleration)."""
    pr = {}
    for f in ("gid", "class_owner", "sub_index"):
        pr[f] = p1[f]
    pr["hlc"] = p1["hlc"] + (p1["hlc"] - p2["hlc"] if p2 else 0)
    for f in ("posx", "posy", "posz", "swing_x", "swing_z", "twist"):
        pr[f] = to_range(p1[f] + (p1[f] - p2[f] if p2 else 0), f)
    return pr


def predict_derived(p1, cur, k):
    """Velocity and acceleration from the position and velocity already decoded: mm_to_packets' own rule."""
    vel = clamp(int(round((cur[f"pos{k}"] - p1[f"pos{k}"]) / V_MAX * 32767)))
    return vel


def encode_slot(raw, p1, p2):
    if raw[MARK_OFF] == MARK:
        raise ValueError("a source packet cannot carry the marker byte (class 0xFF is not a class)")
    if p1 is None:
        return raw
    cur = vals(raw)
    pr = predict(p1, p2, cur)
    for k in "xyz":  # derived: velocity from this position, acceleration from this velocity
        pr[f"vel{k}"] = predict_derived(p1, cur, k)
        pr[f"acc{k}"] = clamp(int(round((cur[f"vel{k}"] - p1[f"vel{k}"]) * V_MAX / 32767 / A_MAX * 32767)))
    out = bytearray(SIZE)
    esc = bytearray()
    for i, f in enumerate(FIELDS):
        r = wrap(cur[f] - pr[f], MOD[f])
        if -127 <= r <= 127:
            out[i] = r & 0xFF
            continue
        w = WIDTH[f]
        if not -(1 << (8 * w - 1)) <= r < (1 << (8 * w - 1)):
            return raw  # the residual does not fit its escape slot: send the packet as it is
        out[i] = ESC & 0xFF
        esc += r.to_bytes(w, "little", signed=True)
    if len(esc) > len(ESC_AT):
        return raw
    for b, at in zip(esc, ESC_AT):
        out[at] = b
    out[MARK_OFF] = MARK
    for j in range(36):
        out[52 + j] = raw[62 + j] ^ p1["_raw"][62 + j]
    for j in range(2):
        out[88 + j] = raw[98 + j] ^ p1["_raw"][98 + j]
    return bytes(out)


def decode_slot(pkt, p1, p2):
    if len(pkt) != SIZE:
        raise ValueError(f"a packet is {SIZE} bytes, got {len(pkt)}")
    if pkt[MARK_OFF] != MARK:
        return pkt
    if p1 is None:
        raise ValueError("a residual packet with no earlier packet in its slot")
    pr = predict(p1, p2, {})
    esc_at = iter(ESC_AT)
    cur = {}
    for i, f in enumerate(FIELDS):
        if f.startswith("vel"):
            pr[f] = predict_derived(p1, cur, f[3])
        elif f.startswith("acc"):
            k = f[3]
            pr[f] = clamp(int(round((cur[f"vel{k}"] - p1[f"vel{k}"]) * V_MAX / 32767 / A_MAX * 32767)))
        s = struct.unpack_from("<b", pkt, i)[0]
        if s == ESC:
            w = WIDTH[f]
            buf = bytes(pkt[next(esc_at)] for _ in range(w))
            r = int.from_bytes(buf, "little", signed=True)
        else:
            r = s
        cur[f] = to_range(pr[f] + r, f)
    p = decode(bytes(SIZE))
    p.gid, p.class_owner, p.sub_index, p.hlc = cur["gid"], cur["class_owner"], cur["sub_index"], cur["hlc"]
    p.pos_um_x, p.pos_um_y, p.pos_um_z = cur["posx"], cur["posy"], cur["posz"]
    p.vel_x, p.vel_y, p.vel_z = cur["velx"], cur["vely"], cur["velz"]
    p.accel_x, p.accel_y, p.accel_z = cur["accx"], cur["accy"], cur["accz"]
    p.swing_x, p.swing_z, p.twist = cur["swing_x"], cur["swing_z"], cur["twist"]
    p.payload = bytes(pkt[52 + j] ^ p1["_raw"][62 + j] for j in range(36))
    b = bytearray(encode(p))
    for j in range(2):
        b[98 + j] = pkt[88 + j] ^ p1["_raw"][98 + j]
    return bytes(b)


def remember(history, e, raw):
    v = vals(raw)
    v["_raw"] = raw
    h = history.get(e, [None, None])
    history[e] = [v, h[0]]


def main():
    path = sys.argv[1]
    ent = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    data = open(path, "rb").read()
    ticks = min(nt, len(data) // (ent * SIZE))
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + label + (f"  [{detail}]" if detail else ""), flush=True)

    enc_h, dec_h = {}, {}
    wire, bad, verbatim, total = [], 0, 0, 0
    for t in range(ticks):
        tick = bytearray()
        for e in range(ent):
            raw = data[(t * ent + e) * SIZE:(t * ent + e + 1) * SIZE]
            h = enc_h.get(e, [None, None])
            pkt = encode_slot(raw, h[0], h[1])
            d = dec_h.get(e, [None, None])
            back = decode_slot(pkt, d[0], d[1])
            bad += back != raw
            verbatim += pkt[MARK_OFF] != MARK
            total += 1
            remember(enc_h, e, raw)
            remember(dec_h, e, back)
            tick += pkt
        wire.append(bytes(tick))
    check("GOLDEN: every packet comes back byte for byte", bad == 0, f"{bad} of {total} differ")
    check("INVARIANT: every coded packet is 100 bytes", all(len(w) == ent * SIZE for w in wire))
    print(f"INFO verbatim (first packet of a slot or a residual that did not fit): {verbatim} of {total} = {verbatim/total*100:.2f}%")
    zero = sum(w.count(0) for w in wire) / sum(len(w) for w in wire)

    def with_prefix(frames):
        tot = 0
        for i in range(1, len(frames)):
            d = zstd.ZstdCompressionDict(frames[i - 1], dict_type=zstd.DICT_TYPE_RAWCONTENT)
            tot += len(zstd.ZstdCompressor(level=1, dict_data=d).compress(frames[i]))
        return tot / (len(frames) - 1)

    old = [data[t * ent * SIZE:(t + 1) * ent * SIZE] for t in range(ticks)]
    a, b = with_prefix(wire), with_prefix(old)
    print(f"SIZE {zero*100:.1f}% of coded bytes are zero")
    print(f"SIZE coded: {a:.0f} B/tick = {a/ent*8:.1f} bits/entity    old layout: {b:.0f} B/tick = {b/ent*8:.1f} bits/entity (zstd L1, previous tick as prefix)")
    check("smaller than the old layout under the same zstd", a < b, f"{a:.0f} vs {b:.0f}")

    # CONTROLS: each planted defect must be caught
    h = enc_h
    raw = data[(ticks - 1) * ent * SIZE:(ticks - 1) * ent * SIZE + SIZE]
    p1 = vals(data[(ticks - 2) * ent * SIZE:(ticks - 2) * ent * SIZE + SIZE])
    p1["_raw"] = data[(ticks - 2) * ent * SIZE:(ticks - 2) * ent * SIZE + SIZE]
    pkt = bytearray(encode_slot(raw, p1, None))
    if pkt[MARK_OFF] == MARK:
        pkt[4] ^= 0x03  # one residual byte
        check("CONTROL: a flipped residual byte is caught", decode_slot(bytes(pkt), p1, None) != raw)
    odd = bytearray(raw)
    odd[98], odd[40] = 0x5A, odd[40] ^ 0x7F  # a non-zero reserved byte and an odd HLC: still exact
    pkt = encode_slot(bytes(odd), p1, None)
    check("CONTROL: a non-zero reserved byte and a jumped HLC still roundtrip", decode_slot(pkt, p1, None) == bytes(odd))
    try:
        encode_slot(bytes(raw[:51]) + bytes([MARK]) + bytes(raw[52:]), p1, None)
        check("CONTROL: a source packet carrying the marker is refused", False)
    except ValueError:
        check("CONTROL: a source packet carrying the marker is refused", True)
    try:
        decode_slot(raw[:-1], p1, None)
        check("CONTROL: a short packet is refused", False)
    except ValueError:
        check("CONTROL: a short packet is refused", True)
    print("RESULT", "ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


main()
