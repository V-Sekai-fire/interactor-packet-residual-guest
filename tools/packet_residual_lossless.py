# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT

# Spike for issue #5: the current 100-byte entity packet, coded LOSSLESSLY as predicted + residual fix.
#   PACKET_MODULE_DIR=... PREDICT_GOLDEN=.../predict_golden.csv python packet_residual_lossless.py FRAMES.bin [ENTITIES=240] [TICKS=400]
# FRAMES.bin is ticks x ENTITIES x 100 B (mm_to_packets.py). Every byte of every packet comes back exact: gid,
# class_owner, sub_index, hlc, position, velocity, acceleration, rotation, payload and the reserved tail.
#
# A coded packet is still exactly 100 bytes, whole or malformed, in one of two forms told apart by byte 51 (the class
# byte of class_owner, little endian): a listed class (0..7, 128) is the old packet verbatim; 0xFF marks a residual
# packet. No source packet can carry 0xFF (the codec drops an unlisted class), so nothing is ambiguous and nothing is
# refused: a packet whose residuals do not fit takes the verbatim form.
#
# Residual packet:  0..15   stage 1, one int8 per scalar (0 = the prediction was exact, -128 = escape)
#                   16..50, 90..99   stage 2, the escaped residuals in decode order (u32/i32 fields 4 B, i16 fields 2 B)
#                   51      marker 0xFF
#                   52..87  payload XOR the previous payload      88..89  reserved XOR the previous reserved
# Everything not written is zero, so zstd and run-length coding on the wire make it nearly free.
#
# Two predictors, chosen per packet from the slot's history, so both ends choose alike:
#   deadreck  position from the last two packets; velocity from this position (mm_to_packets' rule).
#   kernel    the contract's Makima kernel (predict_kernel.py, golden against predict_golden.csv): velocity from its own
#             history first, then position from the three previous packets' velocity plus this packet's velocity.
#             It needs three earlier packets in the slot; with fewer the packet uses deadreck.
import os
import struct
import sys

import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The packet module is emitted from contract-entity-packet (Lean) and is not copied here: point PACKET_MODULE_DIR at
# the directory that holds xr_grid_entity_packet.py (mixer-broadcaster keeps one at clients/python/tools).
sys.path.insert(0, os.environ.get("PACKET_MODULE_DIR", os.path.dirname(os.path.abspath(__file__))))
try:
    from xr_grid_entity_packet import SIZE, decode, encode  # noqa: E402
except ImportError:
    sys.exit("set PACKET_MODULE_DIR to the directory holding xr_grid_entity_packet.py (emitted from contract-entity-packet)")
import predict_kernel  # noqa: E402

V_MAX, A_MAX = 500000, 1000000
MARK, MARK_OFF = 0xFF, 51
ESC = -128
ESC_AT = list(range(16, 51)) + list(range(90, 100))
# scalar fields in wire (slot) order
FIELDS = (
    ["gid", "class_owner", "sub_index", "hlc"]
    + [f"pos{k}" for k in "xyz"] + [f"vel{k}" for k in "xyz"] + [f"acc{k}" for k in "xyz"]
    + ["swing_x", "swing_z", "twist"]
)
WIDTH = {f: 2 if f.startswith(("vel", "acc", "swing", "twist")) else 4 for f in FIELDS}
MOD = {f: (1 << 16) if WIDTH[f] == 2 else (1 << 64) for f in FIELDS}
for f in ("gid", "class_owner", "sub_index"):
    MOD[f] = 1 << 32
MOD["hlc"] = 1 << 64
SIGNED = {f: f.startswith(("pos", "vel", "acc", "swing", "twist")) for f in FIELDS}
HEAD = ["gid", "class_owner", "sub_index", "hlc"]
ROT = ["swing_x", "swing_z", "twist"]
POS, VEL, ACC = [f"pos{k}" for k in "xyz"], [f"vel{k}" for k in "xyz"], [f"acc{k}" for k in "xyz"]
ORDER = {"deadreck": HEAD + POS + VEL + ACC + ROT, "kernel": HEAD + VEL + POS + ACC + ROT}
PLANT_UM = 0  # a planted position error in the kernel predictor, to prove the cost sees the predictor


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


def mode_for(h, want):
    """The predictor this packet uses: kernel only when three earlier packets of the slot exist."""
    return "kernel" if want == "kernel" and h[2] is not None else "deadreck"


def predict(f, h, cur, mode):
    """Prediction of scalar f. h = the slot's earlier exact packets, newest first (None where missing); cur = the
    fields of this packet already known, in ORDER[mode]."""
    p1, p2 = h[0], h[1]
    if f in ("gid", "class_owner", "sub_index"):
        return p1[f]
    if f == "hlc":
        return p1["hlc"] + (p1["hlc"] - p2["hlc"] if p2 else 0)
    if f in ROT:
        return to_range(p1[f] + (p1[f] - p2[f] if p2 else 0), f)
    k = f[-1]
    if f in POS:
        if mode == "kernel":
            d = predict_kernel.predict_disp_um(h[2]["vel" + k], h[1]["vel" + k], h[0]["vel" + k], cur["vel" + k], 1, V_MAX)
            return p1[f] + d + PLANT_UM * ((p1[f] % 5) - 2)  # varying, so the plant cannot compress away
        return to_range(p1[f] + (p1[f] - p2[f] if p2 else 0), f)
    if f in VEL:
        if mode == "kernel":  # its own history: constant acceleration
            return clamp(p1[f] + (p1[f] - p2[f]))
        return clamp(int(round((cur["pos" + k] - p1["pos" + k]) / V_MAX * 32767)))
    return clamp(int(round((cur["vel" + k] - p1["vel" + k]) * V_MAX / 32767 / A_MAX * 32767)))  # acc


def encode_slot(raw, h, want="kernel"):
    if raw[MARK_OFF] == MARK:
        raise ValueError("a source packet cannot carry the marker byte (class 0xFF is not a class)")
    if h[0] is None:
        return raw
    mode = mode_for(h, want)
    cur = vals(raw)
    out = bytearray(SIZE)
    esc = bytearray()
    seen = {}
    for f in ORDER[mode]:
        r = wrap(cur[f] - predict(f, h, seen, mode), MOD[f])
        seen[f] = cur[f]
        i = FIELDS.index(f)
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
        out[52 + j] = raw[62 + j] ^ h[0]["_raw"][62 + j]
    for j in range(2):
        out[88 + j] = raw[98 + j] ^ h[0]["_raw"][98 + j]
    return bytes(out)


def decode_slot(pkt, h, want="kernel"):
    if len(pkt) != SIZE:
        raise ValueError(f"a packet is {SIZE} bytes, got {len(pkt)}")
    if pkt[MARK_OFF] != MARK:
        return pkt
    if h[0] is None:
        raise ValueError("a residual packet with no earlier packet in its slot")
    mode = mode_for(h, want)
    esc_at = iter(ESC_AT)
    cur = {}
    for f in ORDER[mode]:
        s = struct.unpack_from("<b", pkt, FIELDS.index(f))[0]
        if s == ESC:
            r = int.from_bytes(bytes(pkt[next(esc_at)] for _ in range(WIDTH[f])), "little", signed=True)
        else:
            r = s
        cur[f] = to_range(predict(f, h, cur, mode) + r, f)
    p = decode(bytes(SIZE))
    p.gid, p.class_owner, p.sub_index, p.hlc = cur["gid"], cur["class_owner"], cur["sub_index"], cur["hlc"]
    p.pos_um_x, p.pos_um_y, p.pos_um_z = cur["posx"], cur["posy"], cur["posz"]
    p.vel_x, p.vel_y, p.vel_z = cur["velx"], cur["vely"], cur["velz"]
    p.accel_x, p.accel_y, p.accel_z = cur["accx"], cur["accy"], cur["accz"]
    p.swing_x, p.swing_z, p.twist = cur["swing_x"], cur["swing_z"], cur["twist"]
    p.payload = bytes(pkt[52 + j] ^ h[0]["_raw"][62 + j] for j in range(36))
    b = bytearray(encode(p))
    for j in range(2):
        b[98 + j] = pkt[88 + j] ^ h[0]["_raw"][98 + j]
    return bytes(b)


def remember(history, e, raw, coded):
    """A verbatim packet restarts the slot's history on both ends; a residual packet extends it. A keyframe is a verbatim packet."""
    v = vals(raw)
    v["_raw"] = raw
    h = history.get(e, [None, None, None])
    history[e] = [v, h[0], h[1]] if coded[MARK_OFF] == MARK else [v, None, None]


def run(data, ent, ticks, want):
    """Encode every slot, decode it with a separate state, compare. Returns (wire frames, bad, verbatim, total, nz)."""
    enc_h, dec_h, wire = {}, {}, []
    bad = verbatim = total = n_res = 0
    nz = {f: 0 for f in FIELDS}
    for t in range(ticks):
        tick = bytearray()
        for e in range(ent):
            raw = data[(t * ent + e) * SIZE:(t * ent + e + 1) * SIZE]
            h = enc_h.get(e, [None, None, None])
            pkt = encode_slot(raw, h, want)
            back = decode_slot(pkt, dec_h.get(e, [None, None, None]), want)
            bad += back != raw
            total += 1
            if pkt[MARK_OFF] == MARK:
                n_res += 1
                for i, f in enumerate(FIELDS):
                    nz[f] += pkt[i] != 0
            else:
                verbatim += 1
            remember(enc_h, e, raw, pkt)
            remember(dec_h, e, back, pkt)
            tick += pkt
        wire.append(bytes(tick))
    return wire, bad, verbatim, total, {f: nz[f] / max(1, n_res) for f in FIELDS}


def with_prefix(frames):
    tot = 0
    for i in range(1, len(frames)):
        d = zstd.ZstdCompressionDict(frames[i - 1], dict_type=zstd.DICT_TYPE_RAWCONTENT)
        tot += len(zstd.ZstdCompressor(level=1, dict_data=d).compress(frames[i]))
    return tot / (len(frames) - 1)


def main():
    global PLANT_UM
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

    gold = os.environ.get("PREDICT_GOLDEN")
    if not gold or not os.path.exists(gold):
        check("GOLDEN: the kernel port matches the contract's predict_golden.csv", False, "set PREDICT_GOLDEN to that file")
    else:
        n, bad = predict_kernel.main(gold)
        check("GOLDEN: the kernel port matches the contract's predict_golden.csv", bad == 0, f"{bad} of {n} differ")

    old = [data[t * ent * SIZE:(t + 1) * ent * SIZE] for t in range(ticks)]
    base = with_prefix(old)
    cost = {}
    for want in ("deadreck", "kernel"):
        wire, bad, verbatim, total, nz = run(data, ent, ticks, want)
        check(f"GOLDEN ({want}): every packet comes back byte for byte", bad == 0, f"{bad} of {total} differ")
        check(f"INVARIANT ({want}): every coded packet is 100 bytes", all(len(w) == ent * SIZE for w in wire))
        cost[want] = with_prefix(wire)
        if want == "kernel" and os.environ.get("DUMP_CODED"):  # the coded stream, for the native test to compare against
            open(os.environ["DUMP_CODED"], "wb").write(b"".join(wire))
        top = ", ".join(f"{f} {nz[f]*100:.1f}%" for f in sorted(nz, key=lambda name: nz[name], reverse=True)[:5])
        print(f"INFO ({want}) verbatim {verbatim/total*100:.2f}%; non-zero residual, top: {top}")
        print(f"SIZE ({want}): {cost[want]:.0f} B/tick = {cost[want]/ent*8:.1f} bits/entity   (old layout {base:.0f} B/tick = {base/ent*8:.1f})")
    print(f"INFO bytes per tick: dead reckoning {cost['deadreck']:.0f}, kernel {cost['kernel']:.0f}")

    # CONTROLS
    PLANT_UM = 1  # the kernel off by one micrometre on both ends: still lossless, and it must cost more
    wire, bad, _, total, _ = run(data, ent, min(ticks, 120), "kernel")
    c_plant = with_prefix(wire)
    PLANT_UM = 0
    wire, _, _, _, _ = run(data, ent, min(ticks, 120), "kernel")
    c_clean = with_prefix(wire)
    check("CONTROL: a planted 1 um kernel error still roundtrips", bad == 0, f"{bad} of {total} differ")
    print(f"INFO planted kernel error {c_plant:.0f} B/tick, clean {c_clean:.0f} B/tick (cost is not a goal; lossless is)")
    raws = [old[t][0:SIZE] for t in range(4)]
    h2 = [vals(raws[2]) | {"_raw": raws[2]}, vals(raws[1]) | {"_raw": raws[1]}, None]
    raw = raws[3]
    pkt = bytearray(encode_slot(raw, h2, "deadreck"))
    if pkt[MARK_OFF] == MARK:
        pkt[4] ^= 0x03
        check("CONTROL: a flipped residual byte is caught", decode_slot(bytes(pkt), h2, "deadreck") != raw)
    odd = bytearray(raw)
    odd[98], odd[40] = 0x5A, odd[40] ^ 0x7F
    check("CONTROL: a non-zero reserved byte and a jumped HLC still roundtrip",
          decode_slot(encode_slot(bytes(odd), h2, "deadreck"), h2, "deadreck") == bytes(odd))
    try:
        encode_slot(bytes(raw[:51]) + bytes([MARK]) + bytes(raw[52:]), h2, "deadreck")
        check("CONTROL: a source packet carrying the marker is refused", False)
    except ValueError:
        check("CONTROL: a source packet carrying the marker is refused", True)
    try:
        decode_slot(raw[:-1], h2, "deadreck")
        check("CONTROL: a short packet is refused", False)
    except ValueError:
        check("CONTROL: a short packet is refused", True)
    print("RESULT", "ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


main()
