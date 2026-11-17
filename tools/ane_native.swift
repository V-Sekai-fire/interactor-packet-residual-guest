// SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
//
// SPDX-License-Identifier: MIT
// Times a Core ML model from native code, no Python: swiftc -O tools/ane_native.swift -o build/ane_native
//   ane_native MODEL.mlmodelc DIMS [REPS]   (DIMS e.g. 1000,256 or 1000,5,128; input float32, first input name; prints us per call for each compute unit)
import CoreML
import Foundation

let a = CommandLine.arguments
guard a.count >= 3 else { exit(2) }
let url = URL(fileURLWithPath: a[1]), dims = a[2].split(separator: ",").map { Int($0)! }, N = dims[0]
let total = dims.reduce(1, *)
let reps = a.count > 3 ? Int(a[3])! : 200
let x = try! MLMultiArray(shape: dims.map { NSNumber(value: $0) }, dataType: .float32)
let p = x.dataPointer.assumingMemoryBound(to: Float.self)
var s: UInt64 = 12345
for i in 0..<total { s = s &* 6364136223846793005 &+ 1442695040888963407; p[i] = Float(Int64(s >> 40) % 4000) / 1000.0 - 2.0 }
for (name, cu) in [("CPU only", MLComputeUnits.cpuOnly), ("CPU+GPU", .cpuAndGPU), ("CPU+ANE", .cpuAndNeuralEngine)] {
    let cfg = MLModelConfiguration(); cfg.computeUnits = cu
    let m = try! MLModel(contentsOf: url, configuration: cfg)
    let key = m.modelDescription.inputDescriptionsByName.keys.first!
    let inp = try! MLDictionaryFeatureProvider(dictionary: [key: MLFeatureValue(multiArray: x)])
    for _ in 0..<20 { _ = try! m.prediction(from: inp) }
    let t0 = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<reps { _ = try! m.prediction(from: inp) }
    let us = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000.0 / Double(reps)
    print(String(format: "N=%5d %-9@ %8.0f us per call (%6.2f us/entity)", N, name as NSString, us, us / Double(N)))
}
