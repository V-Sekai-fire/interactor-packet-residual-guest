# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Where the kinematic bytes go, by field. Each joint packet has residual bytes 4..15 (pos xyz, vel xyz, acc xyz, swing_x, swing_z, twist; -128 marks an escape) and escape bytes 16..50
that hold the full residual of each escaped field in codec order (vel, pos, acc, rot; vel/acc/rot 2 bytes, pos 4). Cost per byte = ideal code length under the static joint x position tables
fitted on the 26 library avatars; each escape byte is charged to the field that escaped. Test avatars, ticks 101 and later. Prints bytes per avatar-tick and how often each field escapes."""
import os, pickle
import numpy as np

T = os.environ["TMPDIR"].rstrip("/")
Wt = np.frombuffer(b"".join(pickle.load(open(T + "/coded_wire_train.pkl", "rb"))), np.uint8).reshape(400, 26, 30, 100)[1:].astype(np.int64)  # our own caches
Wv = np.frombuffer(b"".join(pickle.load(open(T + "/coded_wire.pkl", "rb"))), np.uint8).reshape(400, 8, 30, 100)[101:].astype(np.int64)
cnt = np.full((30, 100, 256), 0.05)
for j in range(30):
    pos = np.broadcast_to(np.arange(100), Wt[:, :, j].shape)
    np.add.at(cnt[j], (pos.ravel(), Wt[:, :, j].ravel()), 1)
lp = np.log2(cnt / cnt.sum(2, keepdims=True))
bits = np.zeros(Wv.shape)
for j in range(30): bits[:, :, j] = -lp[j][np.arange(100)[None, None], Wv[:, :, j]]
FIELD = {4: "pos", 5: "pos", 6: "pos", 7: "vel", 8: "vel", 9: "vel", 10: "acc", 11: "acc", 12: "acc", 13: "rot", 14: "rot", 15: "rot"}
order = [7, 8, 9, 4, 5, 6, 10, 11, 12, 13, 14, 15]  # codec order for the kernel mode: vel, pos, acc, rot
width = {7: 2, 8: 2, 9: 2, 4: 4, 5: 4, 6: 4, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2}
tot = {k: 0.0 for k in ("pos", "vel", "acc", "rot", "escape marker")}; esc_rate = {k: [] for k in ("pos", "vel", "acc", "rot")}
bits_f = {k: np.zeros(Wv.shape[:3]) for k in ("pos", "vel", "acc", "rot")}
for i in range(4, 16): bits_f[FIELD[i]] += bits[..., i]
slot = np.zeros(Wv.shape[:3], int)  # next escape byte
for i in order:
    esc = Wv[..., i] == 128  # -128 as a byte
    esc_rate[FIELD[i]].append(esc.mean())
    for b in range(width[i]):
        sl = slot + b; idx = np.where(sl < 35, 16 + sl, 90 + np.minimum(sl - 35, 9))  # ESC_AT: bytes 16..50 then 90..99
        bits_f[FIELD[i]] += np.where(esc, np.take_along_axis(bits, idx[..., None], 3)[..., 0], 0.0)
    slot = slot + np.where(esc, width[i], 0)
print("bytes per avatar-tick (30 joints), ideal under the static tables, test avatars ticks 101-399:")
for k in ("pos", "vel", "acc", "rot"): print(f"  {k:4s} {bits_f[k].sum(-1).mean() / 8:6.1f}   (a field escapes in {np.mean(esc_rate[k]) * 100:4.1f}% of packets)")
print(f"  sum  {sum(bits_f[k].sum(-1).mean() for k in bits_f) / 8:6.1f};  all 100 bytes of the 30 joints {bits.sum(-1).sum(-1).mean() / 8:6.1f}  (the rest is header, marker, payload bytes)")
