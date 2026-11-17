# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT

"""The ternary layer on the Mac's GPU (Metal) equals the numpy int64 reference bit for bit; GPU and CPU timings beside it.
Run tests/ternary_layer_parity.py first (it builds build/ternary_layer_emit.cpp)."""
import os, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = f"{ROOT}/build"
tmp = tempfile.mkdtemp(prefix="tlm_", dir=os.environ.get("TMPDIR"))
inc = os.path.dirname(os.path.dirname(subprocess.run(["which", "slangc"], capture_output=True, text=True).stdout.strip()))
subprocess.run(["slangc", "-target", "metal", "-stage", "compute", "-entry", "main", "-o", f"{B}/tl.metal", f"{ROOT}/slang/ternary_layer.slang"], check=True, capture_output=True)
subprocess.run(["slangc", "-target", "metal", "-stage", "compute", "-entry", "main", "-o", f"{B}/tlt.metal", f"{ROOT}/slang/ternary_layer_tiled.slang"], check=True, capture_output=True)
for cmd in (["xcrun", "-sdk", "macosx", "metal", "-c", f"{B}/tlt.metal", "-o", f"{B}/tlt.air"],
            ["xcrun", "-sdk", "macosx", "metallib", f"{B}/tlt.air", "-o", f"{B}/tlt.metallib"],
            ["xcrun", "-sdk", "macosx", "metal", "-c", f"{B}/tl.metal", "-o", f"{B}/tl.air"],
            ["xcrun", "-sdk", "macosx", "metallib", f"{B}/tl.air", "-o", f"{B}/tl.metallib"],
            ["swiftc", "-O", f"{ROOT}/tests/ternary_layer_metal.swift", "-o", f"{B}/tl_metal"],
            ["c++", "-O2", "-std=c++17", "-ffp-contract=off", "-Wno-everything", f"-I{ROOT}", f"-I{inc}/include", f"{ROOT}/tests/ternary_layer_harness.cpp", "-o", f"{B}/tl_cpu"]):
    subprocess.run(cmd, check=True, capture_output=True)


def reference(w, scale, bias, x, IN, OUT, BLOCK):
    acc = np.zeros((x.shape[0], OUT), dtype=np.int64)
    for b in range(IN // BLOCK):
        t = x[:, b * BLOCK:(b + 1) * BLOCK].astype(np.int64) @ w[:, b * BLOCK:(b + 1) * BLOCK].astype(np.int64).T
        acc += t * scale[:, b].astype(np.int64)[None]
    return (((acc + 32768) >> 16) + bias[None].astype(np.int64)).astype(np.int32)


def case(w, scale, bias, x, IN, OUT, N, BLOCK):
    p = f"{tmp}/case.bin"
    with open(p, "wb") as f:
        f.write(np.array([IN, OUT, N, BLOCK], dtype=np.int32).tobytes()); f.write(w.tobytes()); f.write(scale.tobytes())
        f.write(bias.tobytes()); f.write(x.tobytes())
    return p


def run_metal(p, N, OUT, reps=1, tiled=False):
    lib = "tlt" if tiled else "tl"
    out = subprocess.run([f"{B}/tl_metal", f"{B}/{lib}.metallib", p, f"{tmp}/ym.bin", str(reps)] + (["tiled"] if tiled else []), capture_output=True, text=True, check=True).stdout
    return np.fromfile(f"{tmp}/ym.bin", dtype=np.int32).reshape(N, OUT), float(out.split()[-1])


def run_cpu(p, reps=1):
    out = subprocess.run([f"{B}/tl_cpu", p, f"{tmp}/yc.bin", str(reps)], capture_output=True, text=True, check=True).stdout
    return float(out.split()[-1])


rng = np.random.default_rng(5)
ok = True
IN, OUT, BLOCK = 256, 256, 128
w = rng.integers(-1, 2, size=(OUT, IN)).astype(np.int8)
scale = rng.integers(1000, 9000, size=(OUT, IN // BLOCK)).astype(np.int32)
bias = rng.integers(-5000, 5000, size=OUT).astype(np.int32)
for N in (240, 1000, 20000):
    x = rng.integers(-(1 << 17), 1 << 17, size=(N, IN)).astype(np.int32)
    want = reference(w, scale, bias, x, IN, OUT, BLOCK)
    p = case(w, scale, bias, x, IN, OUT, N, BLOCK)
    got, _ = run_metal(p, N, OUT)
    bad = int((got != want).sum()); ok &= bad == 0
    _, gpu = run_metal(p, N, OUT, reps=50)
    cpu = run_cpu(p, reps=5 if N <= 1000 else 1)
    gt, _ = run_metal(p, N, OUT, tiled=True)
    badt = int((gt != want).sum()); ok &= badt == 0
    _, gput = run_metal(p, N, OUT, reps=50, tiled=True)
    print(("PASS" if bad == 0 else "FAIL"), f"Metal N={N}: {bad} of {got.size} differ | GPU {gpu:.0f} us per dispatch ({gpu/N:.2f} us/entity), CPU one core {cpu:.0f} us ({cpu/N:.2f} us/entity)")
    print(("PASS" if badt == 0 else "FAIL"), f"Metal tiled N={N}: {badt} of {gt.size} differ | GPU {gput:.0f} us per dispatch ({gput/N:.2f} us/entity)")
w2 = w.copy(); w2[3, 17] = 1 if w2[3, 17] != 1 else -1
x = rng.integers(-(1 << 17), 1 << 17, size=(1000, IN)).astype(np.int32)
want = reference(w, scale, bias, x, IN, OUT, BLOCK)
d = int((run_metal(case(w2, scale, bias, x, IN, OUT, 1000, BLOCK), 1000, OUT)[0] != want).sum()); ok &= d > 0
print(("PASS" if d > 0 else "FAIL"), f"CONTROL one weight flipped on Metal: {d} outputs differ (want > 0)")
dt = int((run_metal(case(w2, scale, bias, x, IN, OUT, 1000, BLOCK), 1000, OUT, tiled=True)[0] != want).sum()); ok &= dt > 0
print(("PASS" if dt > 0 else "FAIL"), f"CONTROL one weight flipped on tiled Metal: {dt} outputs differ (want > 0)")
print("RESULT", "ALL PASS" if ok else "FAILED"); sys.exit(0 if ok else 1)
