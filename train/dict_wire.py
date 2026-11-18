# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""zstd on the real coded wire: prefix of the previous tick, plain levels, and a dictionary trained on four avatars and scored on the other four (and the reverse).
PACKET_MODULE_DIR=... python3 dict_wire.py   (run from tools/; encodes frames.bin once and caches the wire as coded_wire.pkl in TMPDIR)"""
import os, sys, pickle, time
sys.path.insert(0, os.path.expanduser("~/src/interactor-packet-residual-guest/tools"))
import numpy as np, zstandard as zstd
import packet_residual_lossless as P
T = os.environ["TMPDIR"].rstrip("/"); SIZE = P.SIZE; ENT = 240
cache = T + "/coded_wire.pkl"
if os.path.exists(cache):
    wire = pickle.load(open(cache, "rb"))
else:
    data = open(T + "/frames.bin", "rb").read(); t0 = time.time()
    wire, bad, verb, tot, _ = P.run(data, ENT, 400, "kernel"); print("encoded", tot, "packets, bad", bad, f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump(wire, open(cache, "wb"))
print("ticks", len(wire), "wire bytes per tick", len(wire[0]))
def sub(frames, avs):  # the packets of some avatars (30 entities each)
    return [b"".join(f[(a * 30 + e) * SIZE:(a * 30 + e + 1) * SIZE] for a in avs for e in range(30)) for f in frames]
def prefix_cost(fr):
    return sum(len(zstd.ZstdCompressor(level=1, dict_data=zstd.ZstdCompressionDict(fr[i - 1], dict_type=zstd.DICT_TYPE_RAWCONTENT)).compress(fr[i])) for i in range(1, len(fr))) / (len(fr) - 1)
def plain_cost(fr, lvl): return np.mean([len(zstd.ZstdCompressor(level=lvl).compress(f)) for f in fr])
def dict_cost(fr_test, fr_train, size, lvl):
    d = zstd.train_dictionary(size, fr_train)
    return np.mean([len(zstd.ZstdCompressor(level=lvl, dict_data=d).compress(f)) for f in fr_test])
A, B = [0, 1, 2, 3], [4, 5, 6, 7]
res = {}
for name, tr, te in (("train 0-3, score 4-7", A, B), ("train 4-7, score 0-3", B, A)):
    ftr, fte = sub(wire, tr), sub(wire, te)
    res[name] = [prefix_cost(fte), plain_cost(fte, 1), plain_cost(fte, 19)] + [dict_cost(fte, ftr, s, l) for s in (16384, 65536) for l in (1, 19)]
    print(name, "| prefix", round(res[name][0]), "| plain L1", round(res[name][1]), "| plain L19", round(res[name][2]), "| dict 16K L1/L19", round(res[name][3]), round(res[name][4]), "| dict 64K L1/L19", round(res[name][5]), round(res[name][6]), flush=True)
m = np.mean(list(res.values()), 0)
print("MEAN bytes per tick for 120 entities (4 avatars): prefix %.0f | plain L1 %.0f | plain L19 %.0f | dict16K L1 %.0f L19 %.0f | dict64K L1 %.0f L19 %.0f" % tuple(m))
# control: a dictionary trained on the same packets with the byte order of each packet reversed must not help the real stream
ftr = [bytes(b[::-1]) for b in sub(wire, A)]; fte = sub(wire, B)
d = zstd.train_dictionary(65536, ftr); print("control, dictionary trained on byte-reversed packets, L19: %.0f" % np.mean([len(zstd.ZstdCompressor(level=19, dict_data=d).compress(f)) for f in fte]))
