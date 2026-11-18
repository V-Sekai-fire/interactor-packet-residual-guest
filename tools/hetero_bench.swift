// SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
//
// SPDX-License-Identifier: MIT
// Slang ternary layer on every unit, alone and together. One 256 x 256 layer for N entities (a case file as tests/ternary_layer_metal_parity.py writes it).
//   hetero_bench TILED.metallib PLAIN.metallib CASE.bin BLOCK0.mlmodelc BLOCK1.mlmodelc LIBCPU.dylib [SECONDS]
//   GPU tiled  : slang/ternary_layer_tiled.slang -> Metal (a 32-thread group per entity and 4 output rows, one barrier)
//   GPU plain  : slang/ternary_layer.slang       -> Metal (one thread per output)
//   CPU        : slang/ternary_layer.slang       -> C++ (tools/cpu_layer.cpp), CPU_THREADS threads each running a range of thread groups
//   ANE        : the layer's two blocks as Core ML models, 5 digit planes each (no host digit split or recombination: an upper bound)
// Every unit runs the same N entities; a layer is counted when it completes.
import CoreML
import Foundation
import Metal

let a = CommandLine.arguments
guard a.count >= 7, let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { print("usage: see the header"); exit(2) }
let secs = a.count > 7 ? Double(a[7])! : 4.0
let d = try! Data(contentsOf: URL(fileURLWithPath: a[3]))
func i32(_ o: Int) -> Int32 { d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: Int32.self) } }
let IN = Int(i32(0)), OUT = Int(i32(4)), N = Int(i32(8)), BLOCK = Int(i32(12))
var off = 16
func slice(_ n: Int) -> Data { defer { off += n }; return d.subdata(in: off..<off + n) }
let w = slice(OUT * IN), scale = slice(OUT * (IN / BLOCK) * 4), bias = slice(OUT * 4), x = slice(N * IN * 4)
func buf(_ data: Data) -> MTLBuffer { data.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: data.count, options: .storageModeShared)! } }
let cfg = buf(d.subdata(in: 0..<16)), bw = buf(w), bs = buf(scale), bb = buf(bias), bx = buf(x)
let by = dev.makeBuffer(length: N * OUT * 4, options: .storageModeShared)!
func psoOf(_ path: String) -> MTLComputePipelineState {
    try! dev.makeComputePipelineState(function: try! dev.makeLibrary(URL: URL(fileURLWithPath: path)).makeFunction(name: "main_0")!)
}
let psoTiled = psoOf(a[1]), psoPlain = psoOf(a[2])
func gpu(_ pso: MTLComputePipelineState, tiled: Bool) {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso)
    for (i, b) in [cfg, bw, bs, bb, bx, by].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    if tiled { e.dispatchThreadgroups(MTLSize(width: N, height: (OUT + 3) / 4, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)) }
    else { e.dispatchThreadgroups(MTLSize(width: (N * OUT + 63) / 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}
func gpuTiled() { gpu(psoTiled, tiled: true) }
func gpuPlain() { gpu(psoPlain, tiled: false) }

// CPU: the Slang C++ target through a C function; threads split the thread groups of one layer
typealias CpuFn = @convention(c) (UnsafePointer<Int32>, UnsafePointer<Int8>, UnsafePointer<Int32>, UnsafePointer<Int32>, UnsafePointer<Int32>, UnsafeMutablePointer<Int32>, Int64, Int64, UInt32, UInt32, Int64, Int64, Int64) -> Void
let h = dlopen(a[6], RTLD_NOW)!
let cpuFn = unsafeBitCast(dlsym(h, "cpu_layer")!, to: CpuFn.self)
let cfgI: [Int32] = [Int32(IN), Int32(OUT), Int32(N), Int32(BLOCK)]
let wI: [Int8] = w.map { Int8(bitPattern: $0) }
let scI: [Int32] = scale.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
let biI: [Int32] = bias.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
let xI: [Int32] = x.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
let cpuThreads = Int(ProcessInfo.processInfo.environment["CPU_THREADS"] ?? "4")!
let groups = (N * OUT + 63) / 64
let yPtr = UnsafeMutablePointer<Int32>.allocate(capacity: N * OUT)
func cpuLayer() {
    DispatchQueue.concurrentPerform(iterations: cpuThreads) { t in
        let g0 = UInt32(t * groups / cpuThreads), g1 = UInt32((t + 1) * groups / cpuThreads)
        cpuFn(cfgI, wI, scI, biI, xI, yPtr, Int64(N * OUT), Int64(N * IN), g0, g1, Int64(wI.count), Int64(scI.count), Int64(biI.count))
    }
}

let cfgNE = MLModelConfiguration(); cfgNE.computeUnits = .cpuAndNeuralEngine
let m0 = try! MLModel(contentsOf: URL(fileURLWithPath: a[4]), configuration: cfgNE)
let m1 = try! MLModel(contentsOf: URL(fileURLWithPath: a[5]), configuration: cfgNE)
let kdim = 5
let xd = try! MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: kdim), NSNumber(value: BLOCK)], dataType: .float32)
let xp = xd.dataPointer.assumingMemoryBound(to: Float.self)
for i in 0..<(N * kdim * BLOCK) { xp[i] = Float((i * 7919) % 16) - 8 }
let key0 = m0.modelDescription.inputDescriptionsByName.keys.first!, key1 = m1.modelDescription.inputDescriptionsByName.keys.first!
let in0 = try! MLDictionaryFeatureProvider(dictionary: [key0: MLFeatureValue(multiArray: xd)])
let in1 = try! MLDictionaryFeatureProvider(dictionary: [key1: MLFeatureValue(multiArray: xd)])
func ane() { _ = try! m0.prediction(from: in0); _ = try! m1.prediction(from: in1) }

func rate(_ body: @escaping () -> Void) -> Double {  // layers per second for `secs`
    for _ in 0..<3 { body() }
    var n = 0
    let t0 = DispatchTime.now().uptimeNanoseconds
    while Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 < secs { body(); n += 1 }
    return Double(n) / (Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
}
func together(_ units: [(String, () -> Void)]) -> [Double] {
    var r = [Double](repeating: 0, count: units.count)
    let lock = NSLock(), grp = DispatchGroup()
    for (i, u) in units.enumerated() { grp.enter(); DispatchQueue.global().async { let v = rate(u.1); lock.lock(); r[i] = v; lock.unlock(); grp.leave() } }
    grp.wait(); return r
}
func us(_ l: Double) -> String { String(format: "%6.0f layers/s  %.2f us per entity", l, 1e6 / (l * Double(N))) }
let names = ["GPU tiled", "GPU plain", "CPU x\(cpuThreads)", "ANE"]
let U: [(String, () -> Void)] = [("GPU tiled (Slang->Metal)", gpuTiled), ("GPU plain (Slang->Metal)", gpuPlain), ("CPU \(cpuThreads) threads (Slang->C++)", cpuLayer), ("ANE (Core ML, 2 blocks)", ane)]
print("N = \(N) entities per layer, \(secs) s per run")
var alone = [Double]()
for u in U { let v = rate(u.1); alone.append(v); print(u.0.padding(toLength: 36, withPad: " ", startingAt: 0), "alone   ", us(v)) }
for combo in [[0, 3], [1, 3], [0, 2], [0, 2, 3], [0, 1, 2, 3]] {
    let r = together(combo.map { U[$0] }); let tot = r.reduce(0, +), sum = combo.map { alone[$0] }.reduce(0, +)
    print(combo.map { names[$0] }.joined(separator: " + ").padding(toLength: 36, withPad: " ", startingAt: 0), "together", us(tot),
          String(format: " | parts %@ | %.0f%% of the parts alone, %.2fx the fastest", r.map { String(format: "%.0f", $0) }.joined(separator: "+"), 100 * tot / sum, tot / combo.map { alone[$0] }.max()!))
}
