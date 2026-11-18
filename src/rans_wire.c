/*
 * SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
 *
 * SPDX-License-Identifier: MIT
 */

#include "rans_wire.h"

#include <string.h>

#define RANS_L (1u << 23)

int rw_tables_init(rw_tables_t *t, int nctx, const uint16_t *freq, uint16_t *start) {
	for (int c = 0; c < nctx; c++) {
		uint32_t sum = 0;
		for (int s = 0; s < RW_SYMS; s++) {
			uint16_t f = freq[(size_t)c * RW_SYMS + s];
			if (f == 0) return -1;
			start[(size_t)c * (RW_SYMS + 1) + s] = (uint16_t)sum;
			sum += f;
		}
		if (sum != RW_SCALE) return -1;
		start[(size_t)c * (RW_SYMS + 1) + RW_SYMS] = (uint16_t)sum;
	}
	t->nctx = nctx;
	t->freq = freq;
	t->start = start;
	return 0;
}

size_t rw_encode(const rw_tables_t *t, const uint8_t *syms, const uint16_t *ctx, size_t n, uint8_t *out, size_t cap) {
	size_t pos = cap;
	uint32_t x = RANS_L;
	for (size_t i = n; i-- > 0;) {
		uint32_t c = ctx[i], s = syms[i];
		uint32_t fr = t->freq[(size_t)c * RW_SYMS + s], st = t->start[(size_t)c * (RW_SYMS + 1) + s];
		uint32_t xmax = ((RANS_L >> RW_PROB_BITS) << 8) * fr;
		while (x >= xmax) {
			if (pos == 0) return 0;
			out[--pos] = (uint8_t)(x & 0xFF);
			x >>= 8;
		}
		x = ((x / fr) << RW_PROB_BITS) + (x % fr) + st;
	}
	if (pos < 4) return 0;
	for (int k = 0; k < 4; k++) {
		out[--pos] = (uint8_t)(x & 0xFF);
		x >>= 8;
	}
	size_t len = cap - pos;
	memmove(out, out + pos, len);
	return len;
}

int rw_decode(const rw_tables_t *t, const uint8_t *stream, size_t len, const uint16_t *ctx, size_t n, uint8_t *syms) {
	if (len < 4) return -1;
	uint32_t x = ((uint32_t)stream[0] << 24) | ((uint32_t)stream[1] << 16) | ((uint32_t)stream[2] << 8) | stream[3];
	size_t pos = 4;
	if (x < RANS_L) return -1;
	for (size_t i = 0; i < n; i++) {
		uint32_t c = ctx[i];
		if (c >= (uint32_t)t->nctx) return -1;
		const uint16_t *st = t->start + (size_t)c * (RW_SYMS + 1);
		uint32_t slot = x & (RW_SCALE - 1);
		int lo = 0, hi = RW_SYMS - 1;  /* the largest s with st[s] <= slot */
		while (lo < hi) {
			int mid = (lo + hi + 1) >> 1;
			if (st[mid] <= slot) lo = mid; else hi = mid - 1;
		}
		syms[i] = (uint8_t)lo;
		x = (uint32_t)t->freq[(size_t)c * RW_SYMS + lo] * (x >> RW_PROB_BITS) + slot - st[lo];
		while (x < RANS_L) {
			if (pos >= len) return -1;
			x = (x << 8) | stream[pos++];
		}
	}
	return (pos == len && x == RANS_L) ? 0 : -1;
}
