/*
 * SPDX-FileCopyrightText: 2026 K. S. Ernest (iFire) Lee
 *
 * SPDX-License-Identifier: MIT
 */

/* A coded packet is exactly 100 bytes, whole or malformed. Byte 51 (the class byte of class_owner) is 0xFF for a
 * residual packet; any other value is the old packet verbatim, because no source packet can carry 0xFF.
 *
 * The prediction is the contract's Makima kernel (entity_predict.h) with three earlier packets of the slot, dead
 * reckoning with fewer. Both ends compute the same prediction from the same earlier packets, so the residual fix
 * brings back every byte. Plain C, no I/O, no allocation. Built over the contract's emitted headers, which are
 * included from a checkout of contract-entity-packet and are not copied here.
 *
 * Velocity and acceleration are predicted with the rule mm_to_packets.py uses, which is double arithmetic: both ends
 * must run IEEE-754 doubles with no contraction (-ffp-contract=off). The prediction only has to match between the
 * two ends; it does not have to be right. */
#ifndef PACKET_RESIDUAL_H
#define PACKET_RESIDUAL_H

#include <stdint.h>

#include "xr_grid_entity_packet.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PR_SIZE XR_PACKET_SIZE
#define PR_MARK 0xFF
#define PR_MARK_OFF 51

typedef struct {
	xr_grid_entity_packet_t p;
	uint8_t raw[PR_SIZE];
	int valid;
} pr_snap_t;

/* One entity's earlier packets, newest first. Keep one per slot at each end. */
typedef struct {
	pr_snap_t h[3];
} pr_slot_t;

void pr_slot_init(pr_slot_t *s);
/* Remember an exact packet (the source packet at the sender, the decoded one at the receiver). */
void pr_slot_push(pr_slot_t *s, const uint8_t raw[PR_SIZE]);

/* Update a slot after a coded packet, the same call at both ends: a verbatim packet restarts the slot's history (so a keyframe
 * resynchronises a decoder that lost packets), a residual packet extends it. raw is the exact packet. */
void pr_slot_update(pr_slot_t *s, const uint8_t coded[PR_SIZE], const uint8_t raw[PR_SIZE]);
/* A keyframe: the packet verbatim. Returns -1 if the source packet carries the marker byte. */
int pr_keyframe(const uint8_t raw[PR_SIZE], uint8_t out[PR_SIZE]);

/* Code one packet. Returns 1 for a residual packet, 0 for the packet verbatim, -1 if the source packet carries the
 * marker byte (it cannot be coded). */
int pr_encode(const pr_slot_t *s, const uint8_t raw[PR_SIZE], uint8_t out[PR_SIZE]);
/* Bring a coded packet back. Returns 0 on success, -1 for a residual packet with no earlier packet in its slot. */
int pr_decode(const pr_slot_t *s, const uint8_t pkt[PR_SIZE], uint8_t out[PR_SIZE]);

#ifdef __cplusplus
}
#endif
#endif
