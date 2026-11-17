/*
 * SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
 *
 * SPDX-License-Identifier: MIT
 */

#include "packet_residual.h"

#include <math.h>
#include <string.h>

#include "entity_predict.h"

#define NF 16
#define ESC_NEG 0x80
#define V_MAX XR_PACKET_V_MAX_PHYSICAL_DEFAULT_UM_PER_TICK
#define A_MAX XR_PACKET_A_MAX_PHYSICAL_UM_PER_TICK2

/* Scalar fields in slot order. A value is a uint64: the field's bits, signed fields sign-extended. */
enum { GID, CO, SUB, HLC, PX, PY, PZ, VX, VY, VZ, AX, AY, AZ, SWX, SWZ, TW };
static const int BITS[NF] = { 32, 32, 32, 64, 64, 64, 64, 16, 16, 16, 16, 16, 16, 16, 16, 16 };
static const int WIDTH[NF] = { 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
static const int ORDER_DEADRECK[NF] = { GID, CO, SUB, HLC, PX, PY, PZ, VX, VY, VZ, AX, AY, AZ, SWX, SWZ, TW };
static const int ORDER_KERNEL[NF] = { GID, CO, SUB, HLC, VX, VY, VZ, PX, PY, PZ, AX, AY, AZ, SWX, SWZ, TW };
/* The escaped residuals go to bytes 16..50, then 90..99. */
#define ESC_SLOTS 45
static int esc_at(int n) {
	return n < 35 ? 16 + n : 90 + (n - 35);
}

void pr_slot_init(pr_slot_t *s) {
	memset(s, 0, sizeof(*s));
}

void pr_slot_push(pr_slot_t *s, const uint8_t raw[PR_SIZE]) {
	s->h[2] = s->h[1];
	s->h[1] = s->h[0];
	memcpy(s->h[0].raw, raw, PR_SIZE);
	xr_grid_entity_packet_decode(raw, &s->h[0].p);
	s->h[0].valid = 1;
}

void pr_slot_update(pr_slot_t *s, const uint8_t coded[PR_SIZE], const uint8_t raw[PR_SIZE]) {
	if (coded[PR_MARK_OFF] != PR_MARK) {
		pr_slot_init(s);
	}
	pr_slot_push(s, raw);
}

int pr_keyframe(const uint8_t raw[PR_SIZE], uint8_t out[PR_SIZE]) {
	if (raw[PR_MARK_OFF] == PR_MARK) {
		return -1;
	}
	memcpy(out, raw, PR_SIZE);
	return 0;
}

static void fields(const xr_grid_entity_packet_t *p, uint64_t v[NF]) {
	v[GID] = p->gid;
	v[CO] = p->class_owner;
	v[SUB] = p->sub_index;
	v[HLC] = p->hlc;
	v[PX] = (uint64_t)p->pos_um_x;
	v[PY] = (uint64_t)p->pos_um_y;
	v[PZ] = (uint64_t)p->pos_um_z;
	v[VX] = (uint64_t)(int64_t)p->vel_x;
	v[VY] = (uint64_t)(int64_t)p->vel_y;
	v[VZ] = (uint64_t)(int64_t)p->vel_z;
	v[AX] = (uint64_t)(int64_t)p->accel_x;
	v[AY] = (uint64_t)(int64_t)p->accel_y;
	v[AZ] = (uint64_t)(int64_t)p->accel_z;
	v[SWX] = (uint64_t)(int64_t)p->swing_x;
	v[SWZ] = (uint64_t)(int64_t)p->swing_z;
	v[TW] = (uint64_t)(int64_t)p->twist;
}

/* Hold a value the way the field holds it. */
static uint64_t norm(uint64_t x, int f) {
	if (BITS[f] == 64) {
		return x;
	}
	if (BITS[f] == 32) {
		return (uint32_t)x;
	}
	return (uint64_t)(int64_t)(int16_t)(uint16_t)x;
}

/* The signed residual of a difference, wrapped to the field's width. */
static int64_t wrap_bits(uint64_t d, int f) {
	if (BITS[f] == 64) {
		return (int64_t)d;
	}
	if (BITS[f] == 32) {
		return (int64_t)(int32_t)(uint32_t)d;
	}
	return (int64_t)(int16_t)(uint16_t)d;
}

static int64_t clamp32767(double x) {
	int64_t r = (int64_t)x;
	return r > 32767 ? 32767 : r < -32767 ? -32767 : r;
}

static int64_t clampi(int64_t x) {
	return x > 32767 ? 32767 : x < -32767 ? -32767 : x;
}

/* The prediction of field f. v1, v2, v3 = the three earlier packets' fields (v3 = NULL for dead reckoning);
 * cur = this packet's fields already known, in the mode's order. */
static uint64_t predict(int f, const uint64_t *v1, const uint64_t *v2, const uint64_t *v3, const uint64_t cur[NF]) {
	const int kernel = v3 != NULL;
	switch (f) {
		case GID:
		case CO:
		case SUB:
			return v1[f];
		case HLC:
			return v1[HLC] + (v2 ? v1[HLC] - v2[HLC] : 0);
		case SWX:
		case SWZ:
		case TW:
			return norm(v1[f] + (v2 ? v1[f] - v2[f] : 0), f);
		case PX:
		case PY:
		case PZ: {
			if (kernel) {
				const int k = f - PX;
				const int64_t d = xr_predict_disp_um((int32_t)(int64_t)v3[VX + k], (int32_t)(int64_t)v2[VX + k],
						(int32_t)(int64_t)v1[VX + k], (int32_t)(int64_t)cur[VX + k], 1, V_MAX);
				return v1[f] + (uint64_t)d;
			}
			return v1[f] + (v2 ? v1[f] - v2[f] : 0);
		}
		case VX:
		case VY:
		case VZ: {
			if (kernel) {
				return (uint64_t)clampi((int64_t)v1[f] + ((int64_t)v1[f] - (int64_t)v2[f]));
			}
			const int k = f - VX;
			const int64_t dp = (int64_t)(cur[PX + k] - v1[PX + k]);
			return (uint64_t)clamp32767(nearbyint(((double)dp / (double)V_MAX) * 32767.0));
		}
		default: { /* acceleration, from this packet's velocity */
			const int k = f - AX;
			const int64_t dv = (int64_t)cur[VX + k] - (int64_t)v1[VX + k];
			double a = (double)(dv * (int64_t)V_MAX);
			a = a / 32767.0;
			a = a / (double)A_MAX;
			a = a * 32767.0;
			return (uint64_t)clamp32767(nearbyint(a));
		}
	}
}

int pr_encode(const pr_slot_t *s, const uint8_t raw[PR_SIZE], uint8_t out[PR_SIZE]) {
	if (raw[PR_MARK_OFF] == PR_MARK) {
		return -1;
	}
	if (!s->h[0].valid) {
		memcpy(out, raw, PR_SIZE);
		return 0;
	}
	xr_grid_entity_packet_t p;
	xr_grid_entity_packet_decode(raw, &p);
	uint64_t cur[NF], seen[NF], v1[NF], v2[NF], v3[NF];
	fields(&p, cur);
	fields(&s->h[0].p, v1);
	const int has2 = s->h[1].valid, kernel = s->h[2].valid;
	if (has2) {
		fields(&s->h[1].p, v2);
	}
	if (kernel) {
		fields(&s->h[2].p, v3);
	}
	const int *order = kernel ? ORDER_KERNEL : ORDER_DEADRECK;
	uint8_t res[PR_SIZE] = { 0 };
	int n_esc = 0;
	memset(seen, 0, sizeof(seen));
	for (int n = 0; n < NF; ++n) {
		const int f = order[n];
		const int64_t r = wrap_bits(cur[f] - predict(f, v1, has2 ? v2 : NULL, kernel ? v3 : NULL, seen), f);
		seen[f] = cur[f];
		if (r >= -127 && r <= 127) {
			res[f] = (uint8_t)(int8_t)r;
			continue;
		}
		const int w = WIDTH[f];
		if (w == 4 ? (r < INT32_MIN || r > INT32_MAX) : (r < INT16_MIN || r > INT16_MAX)) {
			memcpy(out, raw, PR_SIZE); /* does not fit its escape slot: the packet as it is */
			return 0;
		}
		if (n_esc + w > ESC_SLOTS) {
			memcpy(out, raw, PR_SIZE);
			return 0;
		}
		res[f] = ESC_NEG;
		for (int b = 0; b < w; ++b) {
			res[esc_at(n_esc++)] = (uint8_t)((uint64_t)r >> (8 * b));
		}
	}
	res[PR_MARK_OFF] = PR_MARK;
	for (int j = 0; j < 36; ++j) {
		res[52 + j] = raw[62 + j] ^ s->h[0].raw[62 + j];
	}
	for (int j = 0; j < 2; ++j) {
		res[88 + j] = raw[98 + j] ^ s->h[0].raw[98 + j];
	}
	memcpy(out, res, PR_SIZE);
	return 1;
}

int pr_decode(const pr_slot_t *s, const uint8_t pkt[PR_SIZE], uint8_t out[PR_SIZE]) {
	if (pkt[PR_MARK_OFF] != PR_MARK) {
		memcpy(out, pkt, PR_SIZE);
		return 0;
	}
	if (!s->h[0].valid) {
		return -1;
	}
	uint64_t cur[NF], v1[NF], v2[NF], v3[NF];
	fields(&s->h[0].p, v1);
	const int has2 = s->h[1].valid, kernel = s->h[2].valid;
	if (has2) {
		fields(&s->h[1].p, v2);
	}
	if (kernel) {
		fields(&s->h[2].p, v3);
	}
	const int *order = kernel ? ORDER_KERNEL : ORDER_DEADRECK;
	int n_esc = 0;
	memset(cur, 0, sizeof(cur));
	for (int n = 0; n < NF; ++n) {
		const int f = order[n];
		int64_t r = (int8_t)pkt[f];
		if (pkt[f] == ESC_NEG) {
			const int w = WIDTH[f];
			uint64_t u = 0;
			for (int b = 0; b < w; ++b) {
				u |= (uint64_t)pkt[esc_at(n_esc++)] << (8 * b);
			}
			r = w == 4 ? (int64_t)(int32_t)(uint32_t)u : (int64_t)(int16_t)(uint16_t)u;
		}
		cur[f] = norm(predict(f, v1, has2 ? v2 : NULL, kernel ? v3 : NULL, cur) + (uint64_t)r, f);
	}
	xr_grid_entity_packet_t p;
	memset(&p, 0, sizeof(p));
	p.gid = (uint32_t)cur[GID];
	p.class_owner = (uint32_t)cur[CO];
	p.sub_index = (uint32_t)cur[SUB];
	p.hlc = cur[HLC];
	p.pos_um_x = (int64_t)cur[PX];
	p.pos_um_y = (int64_t)cur[PY];
	p.pos_um_z = (int64_t)cur[PZ];
	p.vel_x = (int16_t)cur[VX];
	p.vel_y = (int16_t)cur[VY];
	p.vel_z = (int16_t)cur[VZ];
	p.accel_x = (int16_t)cur[AX];
	p.accel_y = (int16_t)cur[AY];
	p.accel_z = (int16_t)cur[AZ];
	p.swing_x = (int16_t)cur[SWX];
	p.swing_z = (int16_t)cur[SWZ];
	p.twist = (int16_t)cur[TW];
	for (int j = 0; j < 36; ++j) {
		p.payload[j] = pkt[52 + j] ^ s->h[0].raw[62 + j];
	}
	xr_grid_entity_packet_encode(&p, out);
	for (int j = 0; j < 2; ++j) {
		out[98 + j] = pkt[88 + j] ^ s->h[0].raw[98 + j];
	}
	return 0;
}
