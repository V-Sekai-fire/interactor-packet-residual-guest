/*
 * SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
 *
 * SPDX-License-Identifier: MIT
 */

/* A static rANS coder over bytes, one pinned frequency table per context. Integer only, plain C, no allocation, no I/O.
 *
 * A table row is 256 uint16 frequencies for one context, summing to 1 << RW_PROB_BITS, every byte at least 1, so any byte string codes whatever the tables
 * were fitted on. Both ends hold the same tables (a file pinned by hash); the coder state is a 32-bit value renormalised a byte at a time. The symbols are
 * coded in reverse so the decoder reads forward; the encoder writes the stream backwards into the end of the buffer. This is the coder in
 * train/rans_wire.py, and the two must agree byte for byte. */
#ifndef RANS_WIRE_H
#define RANS_WIRE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RW_PROB_BITS 14
#define RW_SCALE (1u << RW_PROB_BITS)
#define RW_SYMS 256

typedef struct {
	int nctx;
	const uint16_t *freq;  /* nctx * 256, borrowed */
	uint16_t *start;       /* nctx * 257 cumulative starts, owned by the caller */
	uint8_t *bucket;       /* nctx * 256: for each slot >> (RW_PROB_BITS - 8), the first symbol that can hold it; owned by the caller */
} rw_tables_t;

/* Build the cumulative starts (start holds nctx * 257 values; a start is at most 1 << 14, so uint16 holds it). Returns 0, or -1 if a row does not sum to RW_SCALE or holds a zero. */
int rw_tables_init(rw_tables_t *t, int nctx, const uint16_t *freq, uint16_t *start, uint8_t *bucket);

/* Code n symbols, symbol i with context ctx[i]. A symbol costs at most 14 bits, so cap = 2 * n + 8 is always enough.
 * Returns the stream length, or 0 if cap is too small. */
size_t rw_encode(const rw_tables_t *t, const uint8_t *syms, const uint16_t *ctx, size_t n, uint8_t *out, size_t cap);

/* Bring n symbols back. Returns 0 on success, -1 if the stream is malformed (it ends early, has bytes left over, or its start state is out of range). */
int rw_decode(const rw_tables_t *t, const uint8_t *stream, size_t len, const uint16_t *ctx, size_t n, uint8_t *syms);

#ifdef __cplusplus
}
#endif

#endif
