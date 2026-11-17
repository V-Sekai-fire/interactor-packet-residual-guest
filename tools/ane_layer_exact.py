# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""A whole ternary layer (256 -> 256, blocks of 128, 2^16 fixed point, as slang/ternary_layer.slang) with the Neural Engine doing the
dot products. Each activation (a signed integer up to about +-2^19) is split into 5 balanced base-16 digits in [-8, 7]; every digit
matrix is multiplied by a ternary block on the engine (a block sum is at most 128 x 8 = 1024, exact in fp16); the host
recombines the digits and applies the scales in int64. Equal to the numpy int64 reference bit for bit, on every compute unit.
Control: one flipped weight must break it. Prints the model paths for the native timer (tools/ane_native.swift).
Run with the coremltools venv: python tools/ane_layer_exact.py [N]"""
import os, sys, tempfile
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

rng = np.random.default_rng(5)
IN, OUT, BLOCK, K = 256, 256, 128, 5
NB = IN // BLOCK
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
w = rng.integers(-1, 2, size=(OUT, IN)).astype(np.int8)
scale = rng.integers(1000, 9000, size=(OUT, NB)).astype(np.int64)
bias = rng.integers(-5000, 5000, size=OUT).astype(np.int64)
x = rng.integers(-(1 << 17), 1 << 17, size=(N, IN)).astype(np.int64)


def reference(w):
    acc = np.zeros((N, OUT), dtype=np.int64)
    for b in range(NB):
        acc += (x[:, b * BLOCK:(b + 1) * BLOCK] @ w[:, b * BLOCK:(b + 1) * BLOCK].astype(np.int64).T) * scale[:, b][None]
    return ((acc + 32768) >> 16) + bias[None]


def digits(v):  # balanced base 16: v = sum d_k 16^k, d_k in [-8, 7]
    d, r = [], v.copy()
    for _ in range(K):
        dk = ((r + 8) & 15) - 8
        d.append(dk); r = (r - dk) >> 4
    assert not r.any(), "activation outside the digit range"
    return np.stack(d, 1)  # (N, K, IN)


def make_model(wb, path):  # one ternary block (OUT x BLOCK) on the engine: (N, K, BLOCK) digits -> (N, K, OUT) exact integer sums
    @mb.program(input_specs=[mb.TensorSpec(shape=(N, K, BLOCK), dtype=types.fp32)])
    def prog(xd):
        return mb.cast(x=mb.linear(x=mb.cast(x=xd, dtype="fp16"), weight=wb.astype(np.float16)), dtype="fp32")
    ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14, compute_precision=ct.precision.FLOAT16).save(path)


def layer_on(ww, cu, tag):
    dg = digits(x).reshape(N, K, NB, BLOCK)
    tot = np.zeros((N, NB, OUT), dtype=np.int64)
    for b in range(NB):
        p = f"{tmp}/{tag}_b{b}.mlpackage"
        if not os.path.exists(p):
            make_model(ww[:, b * BLOCK:(b + 1) * BLOCK], p)
        m = ct.models.MLModel(p, compute_units=cu)
        key = m.get_spec().description.input[0].name; okey = m.get_spec().description.output[0].name
        o = np.rint(m.predict({key: dg[:, :, b, :].astype(np.float32)})[okey]).astype(np.int64)  # (N, K, OUT)
        tot[:, b] = sum(o[:, k] * (16 ** k) for k in range(K))
    return ((((tot * scale.T[None]).sum(1)) + 32768) >> 16) + bias[None]


tmp = tempfile.mkdtemp(prefix="ane_layer_", dir=os.environ.get("TMPDIR"))
print("models in", tmp)
want = reference(w)
flipped = w.copy(); flipped[3, 17] = 1 if flipped[3, 17] != 1 else -1  # a different weight, so a different layer
ok = True
for tag, ww, must_match in (("trained", w, True), ("control", flipped, False)):
    for name, cu in (("CPU only", ct.ComputeUnit.CPU_ONLY), ("CPU+GPU", ct.ComputeUnit.CPU_AND_GPU), ("CPU+ANE", ct.ComputeUnit.CPU_AND_NE)):
        bad = int((layer_on(ww, cu, tag) != want).sum())
        good = (bad == 0) if must_match else (bad > 0)
        ok &= good
        what = "the trained weights" if must_match else "CONTROL one weight flipped"
        print(("PASS" if good else "FAIL"), f"{what:28s} {name:9s} {bad} of {want.size} outputs differ from the int64 reference")
print("RESULT", "ALL PASS" if ok else "FAILED")
