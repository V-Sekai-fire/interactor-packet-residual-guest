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
- **Neural Engine through Core ML** (`tools/ane_spike.py`, 256-256-90 ternary-weight MLP in fp16): the compute plan puts all 7 ops on the Neural Engine, but the engine computes in fp16. Against a float64 reference, CPU only, CPU+GPU and CPU+ANE are off by up to 27, 2 and 9 rotation codes, and the three units agree on 0 to 9% of values. Each unit repeats itself. So the Neural Engine cannot be the decoder both ends share; it can run a float teacher or a model that is not part of the byte contract.

## Non-goals

- Lossy coding. No field is quantised, rounded or dropped. The decoder returns the current packet byte for byte.
- A cut or truncatable packet. There is no droppable fine stage. A packet is whole or it is malformed.
