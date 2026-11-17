# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT

"""Rung 2: a ternary layer in Slang (C++ target) equals an independent numpy int64 reference, bit for bit.
Controls: one flipped weight, one changed scale and a shifted bias must each break the match."""
import os, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLANGC = os.environ.get("SLANGC", "slangc")
os.makedirs(f"{ROOT}/build", exist_ok=True)
tmp = tempfile.mkdtemp(prefix="tl_", dir=os.environ.get("TMPDIR"))
subprocess.run([SLANGC, "-target", "cpp", "-stage", "compute", "-entry", "main", "-o", f"{ROOT}/build/ternary_layer_emit.cpp",
                f"{ROOT}/slang/ternary_layer.slang"], check=True, capture_output=True)
inc = os.path.dirname(os.path.dirname(subprocess.run(["which", SLANGC], capture_output=True, text=True).stdout.strip()))
exe = f"{tmp}/harness"
r = subprocess.run(["c++", "-O2", "-std=c++17", "-ffp-contract=off", "-Wno-unused-parameter", "-Wno-everything", f"-I{ROOT}", f"-I{inc}/include",
                    f"{ROOT}/tests/ternary_layer_harness.cpp", "-o", exe], capture_output=True, text=True)
if r.returncode:
    print(r.stderr[-1500:]); sys.exit(2)


def reference(w, scale, bias, x, IN, OUT, BLOCK):
    nb = IN // BLOCK
    acc = np.zeros((x.shape[0], OUT), dtype=np.int64)
    for b in range(nb):
        t = x[:, b * BLOCK:(b + 1) * BLOCK].astype(np.int64) @ w[:, b * BLOCK:(b + 1) * BLOCK].astype(np.int64).T  # exact
        acc += t * scale[:, b].astype(np.int64)[None]
    return (((acc + 32768) >> 16) + bias[None].astype(np.int64)).astype(np.int32)


def run(w, scale, bias, x, IN, OUT, N, BLOCK):
    p = f"{tmp}/case.bin"
    with open(p, "wb") as f:
        f.write(np.array([IN, OUT, N, BLOCK], dtype=np.int32).tobytes())
        f.write(w.astype(np.int8).tobytes()); f.write(scale.astype(np.int32).tobytes())
        f.write(bias.astype(np.int32).tobytes()); f.write(x.astype(np.int32).tobytes())
    subprocess.run([exe, p, f"{tmp}/y.bin"], check=True)
    return np.fromfile(f"{tmp}/y.bin", dtype=np.int32).reshape(N, OUT)


rng = np.random.default_rng(5)
ok = True
for IN, OUT, N, BLOCK in [(256, 256, 240, 128), (128, 90, 240, 128), (256, 128, 3, 128), (256, 90, 1000, 128)]:
    w = rng.integers(-1, 2, size=(OUT, IN)).astype(np.int8)
    scale = rng.integers(1000, 9000, size=(OUT, IN // BLOCK)).astype(np.int32)       # about 0.015 to 0.14 in 2^16 fixed point
    bias = rng.integers(-5000, 5000, size=OUT).astype(np.int32)
    x = rng.integers(-(1 << 17), 1 << 17, size=(N, IN)).astype(np.int32)              # up to +-2.0
    want = reference(w, scale, bias, x, IN, OUT, BLOCK)
    got = run(w, scale, bias, x, IN, OUT, N, BLOCK)
    bad = int((got != want).sum())
    ok &= bad == 0
    print(("PASS" if bad == 0 else "FAIL"), f"IN={IN} OUT={OUT} N={N}: {bad} of {got.size} outputs differ")
# controls, on the last case
w2 = w.copy(); w2[3, 17] = -w2[3, 17] if w2[3, 17] else 1
s2 = scale.copy(); s2[5, 0] += 1
b2 = bias.copy(); b2[0] += 1
for name, args in [("one weight flipped", (w2, scale, bias)), ("one scale changed", (w, s2, bias)), ("one bias shifted", (w, scale, b2))]:
    differs = int((run(*args[:3], x, IN, OUT, N, BLOCK) != want).sum())
    ok &= differs > 0
    print(("PASS" if differs > 0 else "FAIL"), f"CONTROL {name}: {differs} outputs differ from the reference (want > 0)")
print("RESULT", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
