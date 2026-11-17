# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Can the Neural Engine compute an integer ternary block exactly? fp16 holds integers exactly up to 2048, so if every partial sum
stays under that, the result is exact in any order and any internal precision. Blocks of 128 ternary weights with activation
digits of at most +-15 sum to at most 1920. Control: activations up to +-127 (sums up to 16256) must break exactness.
Runs the same Core ML model on CPU only, CPU+GPU and CPU+ANE and compares each with a numpy int64 reference.
Run with the coremltools venv: python tools/ane_exact.py"""
import os, sys, tempfile, time
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

rng = np.random.default_rng(9)
BLOCK, NB, OUT = 128, 2, 256
W = rng.integers(-1, 2, size=(OUT, BLOCK)).astype(np.float16)  # ternary, exact in fp16


def build(N, path):
    @mb.program(input_specs=[mb.TensorSpec(shape=(N, NB, BLOCK), dtype=types.fp32)])
    def prog(x):
        h = mb.cast(x=x, dtype="fp16")
        return mb.cast(x=mb.linear(x=h, weight=W), dtype="fp32")  # (N, NB, OUT): one integer dot per block
    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14, compute_precision=ct.precision.FLOAT16)
    m.save(path)


tmp = tempfile.mkdtemp(prefix="ane_exact_", dir=os.environ.get("TMPDIR"))
ok = True
N = 1000
path = f"{tmp}/blk.mlpackage"
build(N, path)
units = [("CPU only", ct.ComputeUnit.CPU_ONLY), ("CPU+GPU", ct.ComputeUnit.CPU_AND_GPU), ("CPU+ANE", ct.ComputeUnit.CPU_AND_NE)]
for label, amp, must_be_exact in [("activations within +-15 (max sum 1920)", 15, True), ("activations within +-127 (max sum 16256), CONTROL", 127, False)]:
    x = rng.integers(-amp, amp + 1, size=(N, NB, BLOCK)).astype(np.float32)
    want = np.einsum("nbi,oi->nbo", x.astype(np.int64), W.astype(np.int64))
    big = int((np.abs(want) > 2048).sum())
    print(f"--- {label}: {big} of {want.size} outputs exceed 2048")
    for name, cu in units:
        m = ct.models.MLModel(path, compute_units=cu)
        key = m.get_spec().description.input[0].name; okey = m.get_spec().description.output[0].name
        got = m.predict({key: x})[okey].astype(np.float64)
        bad = int((got != want).sum())
        verdict = bad == 0
        if must_be_exact:
            ok &= verdict
        print(("PASS" if (verdict if must_be_exact else True) else "FAIL"), f"{name:9s} {bad} of {want.size} outputs differ from the int64 reference" + ("" if must_be_exact else f"  (inexact is expected here; {'exact anyway' if verdict else 'it did break'})"))
print("RESULT", "the engine is exact inside the fp16 integer range" if ok else "NOT exact even inside the fp16 integer range")
