# SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
#
# SPDX-License-Identifier: MIT
"""Spike: the same ternary-weight MLP (256 -> 256 -> 256 -> 90) as a Core ML fp16 model, run on CPU only, CPU+GPU and
CPU+Neural Engine. Questions: does the ANE take it (compute plan), how fast, and is it exact against an integer reference
(it should not be: the engine computes in fp16)."""
import os, sys, time, tempfile
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

rng = np.random.default_rng(3)
IN, H, OUT = 256, 256, 90


def layer(i, o):
    w = rng.integers(-1, 2, size=(o, i)).astype(np.int8)
    s = np.float32(rng.uniform(0.01, 0.06))  # one scale per layer here; per block would only change the fold
    return w, s, (rng.uniform(-0.05, 0.05, size=o)).astype(np.float32)


layers = [layer(IN, H), layer(H, H), layer(H, OUT)]
W16 = [(w.astype(np.float32) * s).astype(np.float16) for w, s, b in layers]  # ternary x scale folded, exactly as fp16 stores it
B16 = [b.astype(np.float16) for w, s, b in layers]


def reference(x):  # float64 math of the fp16 weights: what an ideal engine would give
    a = x.astype(np.float64)
    for k in range(3):
        a = a @ W16[k].astype(np.float64).T + B16[k].astype(np.float64)
        if k < 2:
            a = np.maximum(a, 0)
    return a


def build(N, path):
    @mb.program(input_specs=[mb.TensorSpec(shape=(N, IN), dtype=types.fp16)])
    def prog(x):
        a = x
        for k in range(3):
            a = mb.linear(x=a, weight=W16[k], bias=B16[k], name=f"fc{k}")
            if k < 2:
                a = mb.relu(x=a, name=f"relu{k}")
        return a
    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14, compute_precision=ct.precision.FLOAT16)
    m.save(path)
    return m


tmp = tempfile.mkdtemp(prefix="ane_", dir=os.environ.get("TMPDIR"))
UNITS = [("CPU only", ct.ComputeUnit.CPU_ONLY), ("CPU+GPU", ct.ComputeUnit.CPU_AND_GPU), ("CPU+ANE", ct.ComputeUnit.CPU_AND_NE)]
for N in (240, 1000):
    path = f"{tmp}/mlp{N}.mlpackage"
    build(N, path)
    x = rng.uniform(-2.0, 2.0, size=(N, IN)).astype(np.float16)
    want = reference(x)
    outs = {}
    for name, cu in UNITS:
        m = ct.models.MLModel(path, compute_units=cu)
        key = m.get_spec().description.input[0].name
        okey = m.get_spec().description.output[0].name
        y0 = m.predict({key: x})[okey].astype(np.float64)
        y1 = m.predict({key: x})[okey].astype(np.float64)
        for _ in range(5):
            m.predict({key: x})
        t0 = time.perf_counter()
        R = 60
        for _ in range(R):
            m.predict({key: x})
        us = (time.perf_counter() - t0) / R * 1e6
        err = np.abs(y0 - want) * 32768  # in rotation codes
        outs[name] = y0
        print(f"N={N:5d} {name:9s} {us:8.0f} us per call ({us/N:6.2f} us/entity) | vs fp64 reference: max {err.max():8.1f} codes, mean {err.mean():6.2f} | repeat identical: {np.array_equal(y0, y1)}", flush=True)
    for a, b in [("CPU only", "CPU+GPU"), ("CPU only", "CPU+ANE"), ("CPU+GPU", "CPU+ANE")]:
        print(f"N={N:5d} {a} vs {b}: outputs identical on {np.mean(outs[a]==outs[b])*100:.1f}% of values, max difference {np.abs(outs[a]-outs[b]).max()*32768:.1f} codes")
    if N == 240:
        try:
            from coremltools.models.compute_plan import MLComputePlan
            plan = MLComputePlan.load_from_path(ct.utils.compile_model(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            prog = plan.model_structure.program
            fn = prog.functions["main"]
            devs = {}
            for op in fn.block.operations:
                if op.operator_name in ("const", "constexpr_lut_to_dense"):
                    continue
                u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                d = type(u.preferred_compute_device).__name__ if u else "none"
                devs[d] = devs.get(d, 0) + 1
            print("compute plan, preferred device per op (CPU+ANE allowed):", devs)
        except Exception as e:
            print("compute plan unavailable:", type(e).__name__, str(e)[:160])
