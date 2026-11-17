// SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
//
// SPDX-License-Identifier: MIT
// Do the Neural Engine and the GPU add up? One ternary layer for N entities on each, alone and then both at once on two threads.
//   hetero_bench TL_TILED.metallib CASE.bin BLOCK0.mlmodelc BLOCK1.mlmodelc [SECONDS]
// GPU: the tiled Slang kernel, one dispatch per layer. ANE: two block models (the layer's two blocks of 128), 5 digit planes each, no
// host digit split or recombination (so the ANE figure is an upper bound for the full path). N must equal the case's N and the models'.
import CoreML
import Foundation
import Metal

let a = CommandLine.arguments
guard a.count >= 5, let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { exit(2) }
let secs = a.count > 5 ? Double(a[5])! : 4.0
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

func gpuLayer() {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso)
    for (i, b) in [cfg, bw, bs, bb, bx, by].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.dispatchThreadgroups(MTLSize(width: N, height: (OUT + 3) / 4, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}

let cfgNE = MLModelConfiguration(); cfgNE.computeUnits = .cpuAndNeuralEngine
let m0 = try! MLModel(contentsOf: URL(fileURLWithPath: a[3]), configuration: cfgNE)
let m1 = try! MLModel(contentsOf: URL(fileURLWithPath: a[4]), configuration: cfgNE)
let kdim = 5
let xd = try! MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: kdim), NSNumber(value: BLOCK)], dataType: .float32)
let xp = xd.dataPointer.assumingMemoryBound(to: Float.self)
for i in 0..<(N * kdim * BLOCK) { xp[i] = Float((i * 7919) % 16) - 8 }
let key0 = m0.modelDescription.inputDescriptionsByName.keys.first!, key1 = m1.modelDescription.inputDescriptionsByName.keys.first!
let in0 = try! MLDictionaryFeatureProvider(dictionary: [key0: MLFeatureValue(multiArray: xd)])
let in1 = try! MLDictionaryFeatureProvider(dictionary: [key1: MLFeatureValue(multiArray: xd)])
func aneLayer() { _ = try! m0.prediction(from: in0); _ = try! m1.prediction(from: in1) }

func run(_ name: String, _ body: @escaping () -> Void) -> Double {  // layers per second for `secs`
    for _ in 0..<5 { body() }
    var n = 0
    let t0 = DispatchTime.now().uptimeNanoseconds
    while Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 < secs { body(); n += 1 }
    let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
    return Double(n) / dt
}
let g = run("gpu", gpuLayer)
let n = run("ane", aneLayer)
var rg = 0.0, rn = 0.0
let grp = DispatchGroup()
grp.enter(); DispatchQueue.global().async { rg = run("gpu", gpuLayer); grp.leave() }
grp.enter(); DispatchQueue.global().async { rn = run("ane", aneLayer); grp.leave() }
grp.wait()
func ent(_ l: Double) -> String { String(format: "%7.0f layers/s = %9.0f entities/s", l, l * Double(N)) }
print("N = \(N) entities per layer, \(secs) s per run")
print("GPU alone     ", ent(g))
print("ANE alone     ", ent(n))
print("both at once  ", ent(rg + rn), String(format: " (GPU %.0f + ANE %.0f layers/s)", rg, rn))
print(String(format: "sum of the two alone %.0f layers/s; both at once reach %.0f%% of it, %.2fx the faster one alone", g + n, 100 * (rg + rn) / (g + n), (rg + rn) / max(g, n)))
