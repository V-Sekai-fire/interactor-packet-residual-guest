// SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
//
// SPDX-License-Identifier: MIT

// Runs the Slang ternary layer on the Mac's GPU through Metal.
//   ternary_layer_metal TL.metallib CASE.bin Y.bin [REPS] [tiled]   (CASE format as in ternary_layer_harness.cpp)
// Prints "gpu_us_per_dispatch <n>" for REPS dispatches in one command buffer.
import Foundation
import Metal

let a = CommandLine.arguments
guard a.count >= 4, let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { exit(2) }
let reps = a.count > 4 ? Int(a[4])! : 1
let tiled = a.count > 5 && a[5] == "tiled"
let d = try! Data(contentsOf: URL(fileURLWithPath: a[2]))
func i32(_ o: Int) -> Int32 { d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: Int32.self) } }
let IN = Int(i32(0)), OUT = Int(i32(4)), N = Int(i32(8)), BLOCK = Int(i32(12))
var off = 16
func slice(_ n: Int) -> Data { defer { off += n }; return d.subdata(in: off..<off + n) }
let w = slice(OUT * IN), scale = slice(OUT * (IN / BLOCK) * 4), bias = slice(OUT * 4), x = slice(N * IN * 4)
func buf(_ data: Data) -> MTLBuffer { data.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: data.count, options: .storageModeShared)! } }
let cfg = buf(d.subdata(in: 0..<16)), bw = buf(w), bs = buf(scale), bb = buf(bias), bx = buf(x)
let by = dev.makeBuffer(length: N * OUT * 4, options: .storageModeShared)!
let lib = try! dev.makeLibrary(URL: URL(fileURLWithPath: a[1]))
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "main_0")!)
let cb = q.makeCommandBuffer()!
for _ in 0..<reps {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso)
    for (i, b) in [cfg, bw, bs, bb, bx, by].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    if tiled { e.dispatchThreadgroups(MTLSize(width: N, height: (OUT + 3) / 4, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)) }
    else { e.dispatchThreadgroups(MTLSize(width: (N * OUT + 63) / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)) }
    e.endEncoding()
}
let t0 = DispatchTime.now().uptimeNanoseconds
cb.commit(); cb.waitUntilCompleted()
let t1 = DispatchTime.now().uptimeNanoseconds
print("gpu_us_per_dispatch", Double(t1 - t0) / 1000.0 / Double(reps))
try! Data(bytes: by.contents(), count: N * OUT * 4).write(to: URL(fileURLWithPath: a[3]))
