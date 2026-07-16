/*===- bcir_channel.h - the multi-channel lowering decision in C ------------===
 *
 * The C twin of bcir/channels' routing seam: given a `channel.json` (the channel-plugin
 * manifest, PR #262) and a BCIR claim, decide which channel (backend) the claim targets, so a
 * driver picks its backend with no Python. Two predicates + one pick, mirroring the oracle:
 *   bcir_claim_required_caps  -- the capability tags a claim could be served by  (claim_required_caps)
 *   bcir_channel_suits        -- does a channel suit a claim (declared caps, or the per-kind rule)
 *   bcir_channel_route        -- the cost-free plan-time backend pick among a channel set (route_claim)
 *
 * This is a host tool (it parses JSON + uses libc), not freestanding -- a driver runs it plan-time.
 * A Python<->C parity gate ties it to bcir/channels.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_CHANNEL_H
#define BCIR_CHANNEL_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_cir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The StreamPack-execution capability vocabulary (mirror bcir/channels CAPABILITY_VOCAB). */
#define BCIR_CAP_UNIVERSAL     (1u << 0)   /* runs anything (the CPU fallback)            */
#define BCIR_CAP_DATA_PARALLEL (1u << 1)   /* elementwise / SIMD lanes                    */
#define BCIR_CAP_REDUCE        (1u << 2)   /* reductions / accumulation trees             */
#define BCIR_CAP_GATHER        (1u << 3)   /* random / indexed access (scatter/gather)    */
#define BCIR_CAP_TILE          (1u << 4)   /* blocked tiles (systolic)                    */
#define BCIR_CAP_MATMUL        (1u << 5)   /* dense matmul                                */
#define BCIR_CAP_STREAM_UNIT   (1u << 6)   /* unit-stride sequential streaming            */
#define BCIR_CAP_SCALAR_STREAM (1u << 7)   /* scalar sequential streaming                 */

#define BCIR_CHANNEL_NAME 64
#define BCIR_CHANNEL_KIND 16
#define BCIR_CHANNEL_JSON_MAX_BYTES (1024u * 1024u)

/* A channel, parsed from channel.json (the routing-relevant fields; the K_BCIR cost profile stays
 * on the cost-model rail). */
typedef struct bcir_channel {
  char     name[BCIR_CHANNEL_NAME];
  char     kind[BCIR_CHANNEL_KIND];   /* cpu | gpu | fpga | accelerator | storage | memory */
  uint32_t capabilities;              /* bitmask of BCIR_CAP_* (0 == no declared caps -> legacy) */
  char     provenance[BCIR_CHANNEL_KIND];  /* real | modeled | simulated */
  int      modeled;
} bcir_channel;

/* Parse a channel.json's routing-relevant fields into `ch`. Returns 0 on success, non-zero with a
 * message in `diag`. Duplicate keys, unknown/duplicate capabilities, malformed skipped values,
 * unsupported/control-bearing routing identities, and inputs over
 * BCIR_CHANNEL_JSON_MAX_BYTES are
 * rejected. `ch` is zeroed on every failure. */
int bcir_channel_parse(const char *json, size_t len, bcir_channel *ch, char *diag, size_t dn);

/* The capability tags a claim could be served by (a BCIR_CAP_* bitmask) -- op-derived + stride.
 * Returns 0 for a NULL claim or a claim whose fixed-size op label is not NUL-terminated. */
uint32_t bcir_claim_required_caps(const bcir_claim *cl);

/* Does channel `ch` suit claim `cl`? Declared capabilities (plugin) or the legacy per-kind rule. */
int bcir_channel_suits(const bcir_claim *cl, const bcir_channel *ch);

/* The cost-free plan-time backend pick: the index into channels[0..n) of the chosen channel
 * (most specialized eligible match, tie-broken by name), or -1 if none suits. */
int bcir_channel_route(const bcir_claim *cl, const bcir_channel *channels, int n);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_CHANNEL_H */
