/*===- bcir_q8_tables.h - frozen Q8 memory-tier factors (GENERATED) -------===
 *
 * The K_BCIR memory-tier Q8 factors {bw, lat} vs DRAM (256 == x1.0), baked into the
 * runtime. Source of truth: bcir/kbcir/cost.py MemoryHierarchy.default() (mirrored by
 * the C++ -bcir-cost-model tierForDomain). DO NOT EDIT -- regenerate with
 *   python -m bcir.abi.q8_tables --emit
 * (a drift gate keeps this == a fresh emission). MemTier order, L1=16/16; L2=32/48; L3=96/96; DRAM=256/256; HBM=64/192; CXL=384/512; SSD=1024/4096.
 *
 * C23 #embed (6.10.3) bakes runtime/c/q8_tiers.bin directly when the toolchain
 * supports it; otherwise the identical fallback bytes below (selected by __has_embed).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_Q8_TABLES_H
#define BCIR_Q8_TABLES_H

#include <stdint.h>

#define BCIR_Q8_NTIERS 7
#define BCIR_Q8_ONE 256  /* x1.0 in Q8 */

/* __has_embed must be probed in a NESTED #if: on a pre-#embed toolchain it is not a
 * defined macro, and `defined(__has_embed) && __has_embed(...)` would still try to
 * expand the right operand. The nested form leaves it unparsed when unsupported. */
#if defined(__has_embed)
#  if __has_embed("q8_tiers.bin")
#    define BCIR_Q8_EMBED 1
#  endif
#endif
#if defined(BCIR_Q8_EMBED) && BCIR_Q8_EMBED
static const unsigned char bcir_q8_tiers_blob[] = {
#embed "q8_tiers.bin"
};
#else
#define BCIR_Q8_EMBED 0
static const unsigned char bcir_q8_tiers_blob[] = {
  0x10, 0x00, 0x10, 0x00,
  0x20, 0x00, 0x30, 0x00,
  0x60, 0x00, 0x60, 0x00,
  0x00, 0x01, 0x00, 0x01,
  0x40, 0x00, 0xc0, 0x00,
  0x80, 0x01, 0x00, 0x02,
  0x00, 0x04, 0x00, 0x10,
};
#endif

_Static_assert(sizeof(bcir_q8_tiers_blob) == BCIR_Q8_NTIERS * 4,
               "BCIR Q8 tier table must be NTIERS * {bw,lat} little-endian uint16");

/* MemTier ids (normative; match bcir.kbcir.cost.MemTier / BCIRAttrs.td). */
enum bcir_q8_tier {
  BCIR_Q8_L1 = 0, BCIR_Q8_L2 = 1, BCIR_Q8_L3 = 2, BCIR_Q8_DRAM = 3,
  BCIR_Q8_HBM = 4, BCIR_Q8_CXL = 5, BCIR_Q8_SSD = 6
};

static inline uint16_t bcir_q8_u16(const unsigned char *p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
/* Bandwidth factor (lower = more bandwidth) and latency factor for a tier, Q8 vs DRAM. */
static inline uint16_t bcir_q8_tier_bw(int tier) {
  return bcir_q8_u16(&bcir_q8_tiers_blob[(unsigned)tier * 4 + 0]);
}
static inline uint16_t bcir_q8_tier_lat(int tier) {
  return bcir_q8_u16(&bcir_q8_tiers_blob[(unsigned)tier * 4 + 2]);
}

#endif /* BCIR_Q8_TABLES_H */
