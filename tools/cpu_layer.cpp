// SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
//
// SPDX-License-Identifier: MIT

// The Slang ternary layer on the C++ target, as a C function a host can load: cpu_layer runs thread groups g0..g1 of one layer, so several threads split a layer.
//   c++ -O2 -std=c++17 -ffp-contract=off -Wno-everything -dynamiclib -I. -I<slang>/include tools/cpu_layer.cpp -o build/libcpu_layer.dylib
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <chrono>
namespace tl {
#include "build/ternary_layer_emit.cpp"
}
extern "C" void cpu_layer(const int32_t *cfg, const int8_t *w, const int32_t *scale, const int32_t *bias, const int32_t *x, int32_t *y, int64_t n_out_total, int64_t n_x, uint32_t g0, uint32_t g1, int64_t nw, int64_t ns, int64_t nb) {
	tl::GlobalParams_0 gp{};
	gp.cfg_0 = { const_cast<int32_t *>(cfg), 4 };
	gp.w_0 = { const_cast<int8_t *>(w), (size_t)nw };
	gp.scale_0 = { const_cast<int32_t *>(scale), (size_t)ns };
	gp.bias_0 = { const_cast<int32_t *>(bias), (size_t)nb };
	gp.x_0 = { const_cast<int32_t *>(x), (size_t)n_x };
	gp.y_0 = { y, (size_t)n_out_total };
	tl::ComputeVaryingInput vi{};
	vi.startGroupID = { g0, 0, 0 };
	vi.endGroupID = { g1, 1, 1 };
	tl::main_0(&vi, nullptr, &gp);
}
