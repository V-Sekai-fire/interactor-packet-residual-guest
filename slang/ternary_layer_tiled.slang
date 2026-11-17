/*
 * SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
 *
 * SPDX-License-Identifier: MIT
 */

// The same ternary layer as ternary_layer.slang, scheduled for a GPU: one 32-thread group per (entity, 4 output rows).
// Each thread takes 8 contiguous inputs, loads them once and reuses them for the 4 rows; the 32 partial sums are added in a
// tree. Integer arithmetic, so the result equals the one-thread-per-output kernel bit for bit whatever the order of the sums.
// The grid is (N, ceil(OUT/4), 1). IN must be a multiple of 256 (32 threads x 8 inputs) and BLOCK a multiple of 8.
StructuredBuffer<int> cfg;
StructuredBuffer<int8_t> w;
StructuredBuffer<int> scale;
StructuredBuffer<int> bias;
StructuredBuffer<int> x;
RWStructuredBuffer<int> y;

groupshared int64_t part[4][32];

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 gt : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const int IN = cfg[0], OUT = cfg[1], BLOCK = cfg[3];
    const int e = int(gid.x), o0 = int(gid.y) * 4, lane = int(gt.x);
    const int nb = IN / BLOCK;
    int64_t acc[4] = { 0, 0, 0, 0 };
    for (int base = lane * 8; base < IN; base += 256)
    {
        int xv[8];
        for (int j = 0; j < 8; ++j) xv[j] = x[e * IN + base + j];
        const int b = base / BLOCK;
        for (int r = 0; r < 4; ++r)
        {
            const int o = min(o0 + r, OUT - 1);
            int t = 0;
            for (int j = 0; j < 8; ++j) t += int(w[o * IN + base + j]) * xv[j];
            acc[r] += int64_t(t) * int64_t(scale[o * nb + b]);
        }
    }
    for (int r = 0; r < 4; ++r) part[r][lane] = acc[r];
    for (int s = 16; s > 0; s >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (lane < s)
            for (int r = 0; r < 4; ++r) part[r][lane] += part[r][lane + s];
    }
    if (lane == 0)
        for (int r = 0; r < 4; ++r)
            if (o0 + r < OUT)
                y[e * OUT + o0 + r] = int((part[r][0] + 32768) >> 16) + bias[o0 + r];
}
