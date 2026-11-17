/*
 * SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
 *
 * SPDX-License-Identifier: MIT
 */

// A ternary layer, in integers only: y[e][o] = bias[o] + sum_b scale[o][b] * sum_{i in block b} w[o][i] * x[e][i]
// w is -1, 0 or +1 (int8). x and y are 2^16 fixed point. scale[o][b] is the block's scale in the same fixed point.
// cfg = { IN, OUT, N, BLOCK }. One thread per (entity, output). Integer arithmetic only, so the C++ target, SPIR-V and Metal agree bit for bit.
StructuredBuffer<int> cfg;
StructuredBuffer<int8_t> w;
StructuredBuffer<int> scale;
StructuredBuffer<int> bias;
StructuredBuffer<int> x;
RWStructuredBuffer<int> y;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const int IN = cfg[0], OUT = cfg[1], N = cfg[2], BLOCK = cfg[3];
    const int gid = int(tid.x);
    if (gid >= N * OUT) return;
    const int e = gid / OUT, o = gid % OUT;
    const int nb = IN / BLOCK;
    int64_t acc = 0;
    for (int b = 0; b < nb; ++b)
    {
        int t = 0;
        for (int i = b * BLOCK; i < (b + 1) * BLOCK; ++i)
            t += int(w[o * IN + i]) * x[e * IN + i];
        acc += int64_t(t) * int64_t(scale[o * nb + b]);
    }
    y[e * OUT + o] = int((acc + 32768) >> 16) + bias[o];
}
