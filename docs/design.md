# Design

Both ends hold a shared state S. Each tick each end computes a predicted packet from S. The wire carries the residual R = P - predicted in the zero-run 100-byte packet. The decoder adds R to its own prediction, so the packet is exact for any predictor.

## Layers

| Layer | Predictor | On the wire besides R | Needs bit-exact prediction |
|---|---|---|---|
| L0 (done) | Dead reckoning from the last two packets | nothing | no |
| L1 | Pose looked up from a pick: database hash, clip, frame, root | the pick, in the payload slot | no |
| L2 | A simulator that computes the pick from inputs | inputs repeated in every packet, a state hash about once a second | yes |
| L3 | Tracked joints (VR head and hands) are the residual | corrections for those joints only | no |
| VAE | A small decoder from a few FSQ digits to the pose | the digits | yes |

L2 must produce the same pick that L1 sends: that is its golden test.

## The decoder is pure integer

A float decoder gives a different last bit on a CPU and a GPU, and both ends must predict the same packet. Int8 weights, int32 accumulation and integer requantisation are exact on CPU, Metal and Vulkan. The decoder is a Lean module emitted as Slang. `slangc -target cpp` runs it on the CPU in the realtime loop, with no GPU submit floor. The same source goes to SPIR-V or Metal when there are thousands of avatars, in one dispatch for the whole tick.

## Tests

- GOLDEN: every packet decodes byte for byte.
- INVARIANT: every coded packet is 100 bytes; a short packet is refused.
- CONTROL: a planted wrong predictor still decodes and costs more; a flipped residual byte is caught.
- PARITY (decoder): C++ output equals an independent integer reference equals the GPU output, bit for bit.
- NUMBER: bits per entity against 42.8, and the rotation escape rate.

## Batching

A coded packet is 100 bytes and the wire carries batches of them, back to back. The load of one packet sets how many fit in a batch, so a smaller packet means a larger batch. Measured in bytes per entity of a whole tick under zstd with the previous tick as a prefix, not per packet alone.

A batch also gives a predictor a neighbour: the previous packet in the same batch shares its HLC and has the next gid. That would code the first packet of a slot, which is now sent verbatim, and a late joiner's first batch. It is a later predictor layer, not built.

## Hardware: what can be the shared decoder

The decoder both ends run must give the same bytes on every host, so it is integer. Measured on an Apple M2 Pro, with other jobs running (rough numbers):

- **Metal, tiled integer kernel** (`slang/ternary_layer_tiled.slang`): bit-exact against numpy int64, 0.59 us per entity at 1000 or more entities, 256 x 256 layer.
- **C++ target, one core**: bit-exact, about 5 us per entity for the same layer, not tuned.
- **Neural Engine through Core ML** (`tools/ane_spike.py`, 256-256-90 ternary-weight MLP in fp16): the compute plan puts all 7 ops on the Neural Engine, but the engine computes in fp16. Against a float64 reference, CPU only, CPU+GPU and CPU+ANE are off by up to 27, 2 and 9 rotation codes, and the three units agree on 0 to 9% of values. Each unit repeats itself. Timed from native code with no Python (`tools/ane_native.swift`), at 240 entities the calls take 136 us (CPU), 516 us (CPU+GPU) and 223 us (CPU+ANE); at 1000 entities 334, 805 and 328 us. Python added about 15%, so Core ML's own cost per call is the floor. The fp16 engine is exact when every sum stays inside fp16's exact integers (up to 2048). `tools/ane_exact.py`: a ternary block of 128 with activation digits of at most 15 matches an int64 reference on CPU, GPU and Neural Engine (0 of 512,000 differ); with activations up to 127 all three break at the same 1,613 outputs, so the failure is deterministic too. `tools/ane_layer_exact.py`: a whole 256 x 256 ternary layer, activations split into 5 balanced base-16 digits (each block sum at most 1024), the engine doing the dot products and the host recombining and scaling in int64, equals the numpy reference on all three units (0 of 256,000), and a flipped weight breaks all three. Speed, native timing and other jobs running: one block call at 1000 entities takes 371 us on CPU, 377 us on GPU and 499 us on the Neural Engine, and a layer needs two such calls plus the digit split, against 0.59 us per entity for the whole layer in the tiled Metal kernel. So it can be made exact, and on its own it is not faster here. Run beside the GPU it adds (`tools/hetero_bench.swift`, one 256 x 256 layer for 1000 entities, other jobs running): GPU alone 1,023 layers/s, Neural Engine alone 1,095 (without the host digit split and recombination), both at once 2,100, which is 99% of the sum.
So, with floats, the Neural Engine cannot be the decoder both ends share; it can run a float teacher or a model that is not part of the byte contract.

## Loss

A residual packet is coded against the earlier packets of its slot, so a lost packet leaves the decoder's history different from the encoder's. Measured with no protection (`tests/loss_sim.c`, 240 slots, 400 ticks, one random seed): after one loss the slot decodes wrongly to the end of the run.

Rules now in both codecs:

- A verbatim packet restarts the slot's history at both ends (`pr_slot_update`). A keyframe is a verbatim packet, so it resynchronises a decoder.
- The decoder knows which packets did not arrive. A slot whose history it cannot trust is withheld, never guessed: in every run below, wrong = 0.
- A parity packet is the XOR of the coded packets of G slots in one tick. With one data packet of the group missing and the parity present, the missing packet is rebuilt exactly.
- Keyframes are staggered: slot e sends one when (tick + e) mod K = 0, so the load is flat.

Share of packets delivered exactly, and bytes per tick (240 entities, zstd with the previous tick as prefix; parity and keyframes included):

| G (group) | K (keyframe) | bytes/tick | exact at 1% loss | exact at 5% loss |
|---|---|---|---|---|
| none | none | 1285 | 24.9% | 4.9% |
| none | 20 | 1446 | 90.3% | 61.4% |
| none | 5 | 1673 | 97.1% | 86.0% |
| 8 | none | 1706 | 82.7% | 17.0% |
| 8 | 20 | 2083 | 99.0% | 84.2% |
| 16 | 20 | 1925 | 98.3% | 75.4% |
| 30 | 20 | 1758 | 97.5% | 67.6% |

The old layout costs 1652 bytes per tick. Parity packets are XORs of sparse packets and compress poorly (G=8: 421 bytes per tick). Not measured: parity across a slot's ticks, two parities per group (P+Q), groups chosen across datagrams, bursts of loss.

## Pose to packet code

A joint's rotation goes from its 3x3 matrix to the packet's swing_x, swing_z and twist with no quaternion (`tools/pose_codes.py`): the swing is the shortest turn taking +Y to the matrix's Y column, the twist is what is left. Kimodo's decoder already writes parent-local matrices (`kimodo_decode.h`), so a Kimodo pose reaches the packet by this one step. Checked (`tests/pose_codes_check.py`): the contract's worked examples within one code (the 90 degree swing is the tie 16383.5), 3000 random rotations at 96 microradians worst, and real Kimodo frames at 88; swapped swing axes and a flipped twist sign are caught. Against the quaternion route the dataset was built with, 6 of 17,640 real joint rotations differ, by one code. The dataset keeps its original codes.

## Blocklist

- **A flat MLP over a window of past deltas as the predictor** (operator, 2026-11-17). Measured (`rung3e`, `rung3f`, rotation bytes per tick, same ticks per row, dead reckoning 1080): float 994 at 8 past deltas, 1007 at 32, 1045 at 64, 1059 at 100; ternary 1031, 1033, 1043, 1043. The input grows from 896 to 9,216 values against about 45,000 training windows, and the residual median gets worse as the window grows (27, 29, 36, 38 codes). Use a shape that carries the history in a small state (a recurrent state such as Gated DeltaNet-2, or a short convolution into an integer MLP) and train it with a loss that counts exact hits.

## Non-goals

- Lossy coding. No field is quantised, rounded or dropped. The decoder returns the current packet byte for byte.
- A cut or truncatable packet. There is no droppable fine stage. A packet is whole or it is malformed.
