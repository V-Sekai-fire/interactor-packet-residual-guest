/*
 * SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
 *
 * SPDX-License-Identifier: MIT
 */

/* The C coder against the Python reference. cc -O2 -Isrc src/rans_wire.c tests/rans_test.c -o build/rans_test && build/rans_test DIR
 * DIR holds tables.u16, input.bin (ticks of 12,000 bytes) and streams.bin (a uint32 length and the stream, per tick), all from train/rans_dump.py. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rans_wire.h"

#define TICK 12000
#define NCTX 3000

static uint8_t *slurp(const char *dir, const char *name, size_t *len) {
	char p[1024];
	snprintf(p, sizeof p, "%s/%s", dir, name);
	FILE *f = fopen(p, "rb");
	if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
	fseek(f, 0, SEEK_END); *len = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
	uint8_t *b = malloc(*len);
	if (fread(b, 1, *len, f) != *len) exit(2);
	fclose(f);
	return b;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
	if (argc < 2) return 2;
	size_t nt, ni, ns;
	uint8_t *tb = slurp(argv[1], "tables.u16", &nt), *in = slurp(argv[1], "input.bin", &ni), *st = slurp(argv[1], "streams.bin", &ns);
	uint16_t *freq = malloc(nt);
	for (size_t i = 0; i < nt / 2; i++) freq[i] = (uint16_t)(tb[2 * i] | (tb[2 * i + 1] << 8));
	uint16_t *start = malloc((size_t)NCTX * 257 * 2);
	rw_tables_t t;
	int fails = 0, ok;
	ok = rw_tables_init(&t, NCTX, freq, start) == 0;
	printf("%s tables: every row sums to %u and holds no zero\n", ok ? "PASS" : "FAIL", RW_SCALE); fails += !ok;
	uint16_t ctx[TICK];
	for (int i = 0; i < TICK; i++) ctx[i] = (uint16_t)(i % 3000);  /* four avatars x 30 joints x 100 bytes, in the Python order */
	size_t ticks = ni / TICK, off = 0, same = 0, rt = 0, total = 0;
	uint8_t *buf = malloc(2 * TICK + 8), *back = malloc(TICK);
	double te = 0, td = 0;
	for (size_t k = 0; k < ticks; k++) {
		uint32_t plen = (uint32_t)(st[off] | (st[off + 1] << 8) | (st[off + 2] << 16) | ((uint32_t)st[off + 3] << 24));
		const uint8_t *pstream = st + off + 4;
		off += 4 + plen;
		double a = now();
		size_t len = rw_encode(&t, in + k * TICK, ctx, TICK, buf, 2 * TICK + 8);
		double b = now();
		int r = rw_decode(&t, buf, len, ctx, TICK, back);
		double c = now();
		te += b - a; td += c - b; total += len;
		same += len == plen && memcmp(buf, pstream, len) == 0;
		rt += r == 0 && memcmp(back, in + k * TICK, TICK) == 0;
	}
	ok = same == ticks; printf("%s every tick's stream equals the Python stream byte for byte (%zu of %zu)\n", ok ? "PASS" : "FAIL", same, ticks); fails += !ok;
	ok = rt == ticks; printf("%s every tick decodes back to the exact bytes (%zu of %zu)\n", ok ? "PASS" : "FAIL", rt, ticks); fails += !ok;
	/* controls: a flipped stream byte must be refused or decode to different bytes; a truncated stream and one with an extra byte are refused */
	size_t len = rw_encode(&t, in, ctx, TICK, buf, 2 * TICK + 8);
	buf[len / 2] ^= 0x10;
	int r = rw_decode(&t, buf, len, ctx, TICK, back);
	ok = r != 0 || memcmp(back, in, TICK) != 0; printf("%s CONTROL a flipped stream byte is not decoded to the original\n", ok ? "PASS" : "FAIL"); fails += !ok;
	buf[len / 2] ^= 0x10;
	ok = rw_decode(&t, buf, len - 1, ctx, TICK, back) != 0; printf("%s CONTROL a truncated stream is refused\n", ok ? "PASS" : "FAIL"); fails += !ok;
	buf[len] = 0;
	ok = rw_decode(&t, buf, len + 1, ctx, TICK, back) != 0; printf("%s CONTROL a stream with a byte left over is refused\n", ok ? "PASS" : "FAIL"); fails += !ok;
	/* a byte string the tables never saw still codes: random bytes in */
	uint8_t rnd[TICK];
	srand(7); for (int i = 0; i < TICK; i++) rnd[i] = (uint8_t)(rand() >> 7);
	len = rw_encode(&t, rnd, ctx, TICK, buf, 2 * TICK + 8);
	ok = len && rw_decode(&t, buf, len, ctx, TICK, back) == 0 && memcmp(back, rnd, TICK) == 0; printf("%s random bytes code and decode (%zu bytes for %d)\n", ok ? "PASS" : "FAIL", len, TICK); fails += !ok;
	printf("INFO %zu ticks, mean stream %.1f B per tick; encode %.1f us, decode %.1f us per tick of 4 avatars (%.2f us per packet encode)\n", ticks, (double)total / ticks, te / ticks * 1e6, td / ticks * 1e6, te / ticks * 1e6 / 120);
	return fails ? 1 : 0;
}
