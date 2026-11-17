/* SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
 *
 * SPDX-License-Identifier: MIT
 */
/* Random packet loss, with parity packets and staggered keyframes.
 *   loss_sim FRAMES.bin ENT TICKS G K LOSS_PERCENT SEED [PARITY_OUT.bin [CODED_OUT.bin]]
 * Each tick the slots are grouped G at a time; a parity packet is the XOR of the group's coded packets (G=0: none).
 * Slot e sends a verbatim keyframe at ticks where (t + e) % K == 0 (K=0: never). Every packet, data or parity, is lost
 * with probability LOSS_PERCENT. The decoder knows which packets did not arrive. A group missing one data packet with its
 * parity present is rebuilt exactly; a slot it cannot rebuild is withheld until its next verbatim packet.
 * Counts: exact = delivered and equal to the source, withheld = not delivered, wrong = delivered and different (must be 0). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "packet_residual.h"

static uint64_t rs = 88172645463325252ull;
static double rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }

int main(int argc, char **argv) {
	if (argc < 8) return 2;
	const int ent = atoi(argv[2]), ticks = atoi(argv[3]), G = atoi(argv[4]), K = atoi(argv[5]);
	const double loss = atof(argv[6]) / 100.0;
	rs ^= (uint64_t)atoll(argv[7]) * 0x9E3779B97F4A7C15ull;
	FILE *f = fopen(argv[1], "rb");
	if (!f) return 2;
	size_t n = (size_t)ent * ticks * PR_SIZE;
	uint8_t *src = malloc(n);
	if (fread(src, 1, n, f) != n) return 2;
	fclose(f);
	FILE *po = argc > 8 ? fopen(argv[8], "wb") : NULL, *co = argc > 9 ? fopen(argv[9], "wb") : NULL;
	pr_slot_t *enc = calloc(ent, sizeof(pr_slot_t)), *dec = calloc(ent, sizeof(pr_slot_t));
	int *sync = calloc(ent, sizeof(int));
	uint8_t *coded = malloc((size_t)ent * PR_SIZE);
	int *got = malloc(ent * sizeof(int));
	long exact = 0, withheld = 0, wrong = 0, keyframes = 0, rebuilt = 0, lost_data = 0, lost_par = 0, parity_pkts = 0;
	for (int t = 0; t < ticks; ++t) {
		for (int e = 0; e < ent; ++e) {
			const uint8_t *raw = src + ((size_t)t * ent + e) * PR_SIZE;
			uint8_t *pkt = coded + (size_t)e * PR_SIZE;
			const int key = K > 0 && (t + e) % K == 0;
			if (key) { pr_keyframe(raw, pkt); ++keyframes; }
			else if (pr_encode(&enc[e], raw, pkt) < 0) return 3;
			pr_slot_update(&enc[e], pkt, raw);
			got[e] = rnd() >= loss;
			lost_data += !got[e];
		}
		if (co) fwrite(coded, 1, (size_t)ent * PR_SIZE, co);
		const int groups = G > 0 ? (ent + G - 1) / G : 0;
		for (int g = 0; g < groups; ++g) {
			uint8_t par[PR_SIZE] = { 0 };
			const int lo = g * G, hi = lo + G < ent ? lo + G : ent;
			for (int e = lo; e < hi; ++e) for (int b = 0; b < PR_SIZE; ++b) par[b] ^= coded[(size_t)e * PR_SIZE + b];
			if (po) fwrite(par, 1, PR_SIZE, po);
			++parity_pkts;
			const int par_ok = rnd() >= loss;
			lost_par += !par_ok;
			int missing = 0, which = -1;
			for (int e = lo; e < hi; ++e) if (!got[e]) { ++missing; which = e; }
			if (missing == 1 && par_ok) { /* rebuild from the others and the parity */
				uint8_t rb[PR_SIZE];
				memcpy(rb, par, PR_SIZE);
				for (int e = lo; e < hi; ++e) if (e != which) for (int b = 0; b < PR_SIZE; ++b) rb[b] ^= coded[(size_t)e * PR_SIZE + b];
				if (memcmp(rb, coded + (size_t)which * PR_SIZE, PR_SIZE) == 0) { got[which] = 2; ++rebuilt; } /* the rebuild equals what was sent */
			}
		}
		for (int e = 0; e < ent; ++e) {
			const uint8_t *raw = src + ((size_t)t * ent + e) * PR_SIZE;
			const uint8_t *pkt = coded + (size_t)e * PR_SIZE;
			if (!got[e]) { sync[e] = 0; ++withheld; continue; }
			const int verb = pkt[PR_MARK_OFF] != PR_MARK;
			if (!verb && !sync[e]) { ++withheld; continue; }
			uint8_t back[PR_SIZE];
			if (pr_decode(&dec[e], pkt, back) != 0) { ++withheld; continue; }
			pr_slot_update(&dec[e], pkt, back);
			sync[e] = 1;
			if (memcmp(back, raw, PR_SIZE) == 0) ++exact; else ++wrong;
		}
	}
	if (po) fclose(po);
	if (co) fclose(co);
	const long total = (long)ent * ticks;
	printf("G=%-3d K=%-3d loss=%4.1f%%: exact %6.2f%%  withheld %6.2f%%  wrong %ld | lost data %ld, parity %ld of %ld, rebuilt %ld, keyframes %ld\n",
			G, K, loss * 100, 100.0 * exact / total, 100.0 * withheld / total, wrong, lost_data, lost_par, parity_pkts, rebuilt, keyframes);
	return wrong ? 1 : 0;
}
