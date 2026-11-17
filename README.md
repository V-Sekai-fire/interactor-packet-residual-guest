# interactor-packet-residual-guest

The 100-byte entity packet coded losslessly as a prediction plus a residual fix.

Both ends compute the same predicted packet. The wire carries only what the prediction got wrong, in fixed slots, with zero where it was right. The decoder returns the current packet byte for byte, whatever the predictor. A better predictor only makes the packet smaller.

## Status

A spike. The guest ELF is the target and is not built yet. What is here is the verified codec, written in Python, and the plan.

## What is here

- `tools/packet_residual_lossless.py`: the codec with a pluggable `predict()` (today: dead reckoning, plus the velocity and acceleration rule of `mm_to_packets.py`) and its checks. A coded packet is exactly 100 bytes. Byte 51 is `0xFF` for a residual packet; any other value is the old packet, verbatim. No source packet can carry `0xFF`, so a packet that does not fit is sent as it is and nothing is refused.
- `docs/design.md`: the four predictor layers, why the decoder must be pure integer, and the tests.

## Measured

240 entities (8 avatars x 30 joints), 400 ticks at 20 Hz, from the O3DE motion-matching clips. zstd level 1 with the previous tick as prefix:

| | Bytes per tick | Bits per entity |
|---|---|---|
| Current layout | 1652 | 55.1 |
| This codec, dead-reckoning predictor | 1284 | 42.8 |

96,000 of 96,000 packets decode byte for byte. Rotation carries nearly all of the cost: 70% of rotation residuals are non-zero. Velocity and acceleration residuals are zero because the data was made with the same rule the predictor uses. One data source, one predictor, and no packet loss.

## Data

The paired dataset is `chibifire/datasource-packet-residual` on Hugging Face: motion-capture poses with the packet's rotation codes, split by whole clip into train, validation and test. It is private, because it is derived from third-party clips and its licence status is not established. Train and score any learned predictor on those splits only; the test split holds the clips behind the numbers above.

Snapshot: `e22b943a8e30c34d637b9bebfe165bf6f0ab3246`.

## Reproduce

```
python3 ~/src/mixer-broadcaster/clients/python/tools/mm_to_packets.py MMG_DIR frames.bin --avatars 8 --ticks 400 --hz 20
PACKET_MODULE_DIR=~/src/mixer-broadcaster/clients/python/tools python3 tools/packet_residual_lossless.py frames.bin 240 400
```

`MMG_DIR` is a checkout of `V-Sekai-fire/interactor-motion-matching-guest` with `data/o3de-motionmatching`. The packet module is emitted from `contract-entity-packet` and is not copied here.

## Licence

MIT. See [LICENSE](LICENSE).
