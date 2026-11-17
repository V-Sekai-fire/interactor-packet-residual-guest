/*
 * SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
 *
 * SPDX-License-Identifier: MIT
 */

/* Native check of src/packet_residual.c against the Python codec. A tick is a batch of ENT packets, one per slot,
 * back to back with no framing, as on the wire.
 *   native_test FRAMES.bin CODED.bin [ENT=240] [TICKS=400]
 * FRAMES.bin is the source packets; CODED.bin is what tools/packet_residual_lossless.py wrote (DUMP_CODED). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "packet_residual.h"

static int ok = 1;
static void check(const char *label, int cond, const char *detail) {
	ok &= cond;
	printf("%s %s%s%s\n", cond ? "PASS" : "FAIL", label, detail[0] ? "  " : "", detail);
}

static uint8_t *slurp(const char *path, size_t *n) {
	FILE *f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "cannot open %s\n", path);
		exit(2);
	}
	fseek(f, 0, SEEK_END);
	*n = (size_t)ftell(f);
	fseek(f, 0, SEEK_SET);
	uint8_t *b = malloc(*n);
	if (fread(b, 1, *n, f) != *n) {
		exit(2);
	}
	fclose(f);
	return b;
}

int main(int argc, char **argv) {
	if (argc < 3) {
		fprintf(stderr, "usage: native_test FRAMES.bin CODED.bin [ENT] [TICKS]\n");
		return 2;
	}
	const int ent = argc > 3 ? atoi(argv[3]) : 240;
	int ticks = argc > 4 ? atoi(argv[4]) : 400;
	size_t nsrc, ncod;
	uint8_t *src = slurp(argv[1], &nsrc), *cod = slurp(argv[2], &ncod);
	if ((size_t)ticks * ent * PR_SIZE > nsrc) {
		ticks = (int)(nsrc / ((size_t)ent * PR_SIZE));
	}
	check("the coded stream is as long as the source", ncod >= (size_t)ticks * ent * PR_SIZE, "");
	pr_slot_t *enc = calloc((size_t)ent, sizeof(pr_slot_t)), *dec = calloc((size_t)ent, sizeof(pr_slot_t));
	long total = 0, differ_py = 0, differ_back = 0, residual = 0, rejected = 0;
	for (int t = 0; t < ticks; ++t) {
		for (int e = 0; e < ent; ++e) {
			const uint8_t *raw = src + ((size_t)t * ent + e) * PR_SIZE;
			const uint8_t *want = cod + ((size_t)t * ent + e) * PR_SIZE;
			uint8_t pkt[PR_SIZE], back[PR_SIZE];
			const int r = pr_encode(&enc[e], raw, pkt);
			if (r < 0) {
				++rejected;
				continue;
			}
			residual += r;
			differ_py += memcmp(pkt, want, PR_SIZE) != 0;
			if (pr_decode(&dec[e], pkt, back) != 0) {
				++differ_back;
			} else {
				differ_back += memcmp(back, raw, PR_SIZE) != 0;
			}
			pr_slot_push(&enc[e], raw);
			pr_slot_push(&dec[e], back);
			++total;
		}
	}
	char d[96];
	snprintf(d, sizeof d, "%ld of %ld differ", differ_py, total);
	check("PARITY: the C encoder writes the same bytes as the Python codec", differ_py == 0, d);
	snprintf(d, sizeof d, "%ld of %ld differ", differ_back, total);
	check("GOLDEN: the C decoder brings back every packet byte for byte", differ_back == 0, d);
	snprintf(d, sizeof d, "%ld of %ld", residual, total);
	printf("INFO residual packets %s\n", d);
	check("no source packet was rejected", rejected == 0, "");

	/* CONTROLS */
	pr_slot_t s;
	pr_slot_init(&s);
	uint8_t pkt[PR_SIZE], back[PR_SIZE];
	const uint8_t *raw0 = src, *raw1 = src + (size_t)ent * PR_SIZE;
	pr_slot_push(&s, raw0);
	int made_residual = pr_encode(&s, raw1, pkt) == 1;
	if (made_residual) {
		pkt[4] ^= 0x03;
		pr_decode(&s, pkt, back);
		check("CONTROL: a flipped residual byte is caught", memcmp(back, raw1, PR_SIZE) != 0, "");
	}
	uint8_t marked[PR_SIZE];
	memcpy(marked, raw1, PR_SIZE);
	marked[PR_MARK_OFF] = PR_MARK;
	check("CONTROL: a source packet carrying the marker is refused", pr_encode(&s, marked, pkt) == -1, "");
	pr_slot_t empty;
	pr_slot_init(&empty);
	memset(pkt, 0, PR_SIZE);
	pkt[PR_MARK_OFF] = PR_MARK;
	check("CONTROL: a residual packet with no earlier packet is refused", pr_decode(&empty, pkt, back) == -1, "");
	uint8_t odd[PR_SIZE];
	memcpy(odd, raw1, PR_SIZE);
	odd[98] = 0x5A;
	odd[40] ^= 0x7F;
	pr_encode(&s, odd, pkt);
	pr_decode(&s, pkt, back);
	check("CONTROL: a non-zero reserved byte and a jumped HLC still roundtrip", memcmp(back, odd, PR_SIZE) == 0, "");
	printf("RESULT %s\n", ok ? "ALL PASS" : "FAILED");
	return ok ? 0 : 1;
}
