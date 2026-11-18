# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Dump what the C test needs: the pinned tables (3000 contexts x 256 uint16 frequencies, little endian) fitted on avatars 0-3, the bytes of avatars 4-7 for every tick
from 1, and the streams the Python reference makes for them (a uint32 length before each). Usage: rans_dump.py OUTDIR [TICKS]"""
import os, pickle, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rans_wire as R

out = sys.argv[1]; ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 400
os.makedirs(out, exist_ok=True)
wire = pickle.load(open(R.T + "/coded_wire.pkl", "rb"))  # our own cache, see rans_wire.py
W = np.frombuffer(b"".join(wire), np.uint8).reshape(len(wire), 8, 30, 100)
f, start = R.build_tables(W, [0, 1, 2, 3]); ctxs = R.ctx_list(4)
open(out + "/tables.u16", "wb").write(f.astype("<u2").tobytes())
inp, streams = bytearray(), bytearray()
for t in range(1, ticks):
    syms = W[t, [4, 5, 6, 7]].reshape(-1)
    inp += syms.tobytes(); b = R.encode(syms.tolist(), ctxs, f, start); streams += struct.pack("<I", len(b)) + b
open(out + "/input.bin", "wb").write(inp); open(out + "/streams.bin", "wb").write(streams)
print("ticks", ticks - 1, "input", len(inp), "streams", len(streams))
