// SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
//
// SPDX-License-Identifier: MIT

// Runs the Slang ternary layer on the C++ target. Reads a binary case, writes y.
//   ternary_layer_harness CASE.bin Y.bin   CASE = int32 IN OUT N BLOCK, int8 w[OUT*IN], int32 scale[OUT*IN/BLOCK], bias[OUT], x[N*IN]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>
namespace tl {
#include "build/ternary_layer_emit.cpp"
}
int main(int argc, char **argv) {
	if (argc < 3) return 2;
	FILE *f = fopen(argv[1], "rb");
	if (!f) return 2;
	int32_t cfg[4];
	if (fread(cfg, 4, 4, f) != 4) return 2;
	const int IN = cfg[0], OUT = cfg[1], N = cfg[2], BLOCK = cfg[3];
	std::vector<int8_t> w(OUT * IN);
	std::vector<int32_t> scale(OUT * (IN / BLOCK)), bias(OUT), x(N * IN), y(N * OUT, 0);
	bool ok = fread(w.data(), 1, w.size(), f) == w.size() && fread(scale.data(), 4, scale.size(), f) == scale.size() &&
			fread(bias.data(), 4, bias.size(), f) == bias.size() && fread(x.data(), 4, x.size(), f) == x.size();
	fclose(f);
	if (!ok) return 2;
	tl::GlobalParams_0 gp{};
	gp.cfg_0 = { cfg, 4 };
	gp.w_0 = { w.data(), w.size() };
	gp.scale_0 = { scale.data(), scale.size() };
	gp.bias_0 = { bias.data(), bias.size() };
	gp.x_0 = { x.data(), x.size() };
	gp.y_0 = { y.data(), y.size() };
	tl::ComputeVaryingInput vi{};
	vi.startGroupID = { 0, 0, 0 };
	vi.endGroupID = { (uint32_t)((N * OUT + 63) / 64), 1, 1 };
	const int reps = argc > 3 ? atoi(argv[3]) : 1;
	const auto t0 = std::chrono::steady_clock::now();
	for (int r = 0; r < reps; ++r) tl::main_0(&vi, nullptr, &gp);
	const auto t1 = std::chrono::steady_clock::now();
	printf("cpu_us_per_call %.1f\n", std::chrono::duration<double, std::micro>(t1 - t0).count() / reps);
	f = fopen(argv[2], "wb");
	fwrite(y.data(), 4, y.size(), f);
	fclose(f);
	return 0;
}
