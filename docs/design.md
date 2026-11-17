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

## Non-goals

- Lossy coding. No field is quantised, rounded or dropped. The decoder returns the current packet byte for byte.
- A cut or truncatable packet. There is no droppable fine stage. A packet is whole or it is malformed.
