/*===- bcir_verify.h - the BCIR verifier in C (the C twin of bcir/verify) --===
 *
 * The runnable LangRef laws over the claim graph + plan + StreamPack, in C:
 *   R1-R8   module / claim laws        (bcir_verify_unit, over the claim graph)
 *   R9      K_BCIR plan legality       (bcir_verify_plan, over a bcir_plan)
 *   R10-R11 GEM StreamPack laws        (bcir_verify_pack, over the hydrated bytes)
 *   R12     lowering-contract / support (in bcir_verify_unit)
 *   R13     provenance digest          (bcir_provenance_digest)
 *   R14-R16 CIM/DVFS/alloc smart-lowering -- vacuous for the scalar C subset (no such claims)
 *   R17     accuracy (integer / Q-fixed exact, 0 ULP) -- vacuous for the integer subset
 *   R18     compositional call-graph integrity (in bcir_verify_unit)
 *
 * The verdict logic is pure (IR + plan + runtime status); the diagnostic strings use libc
 * snprintf, so a driver links it host-side or with a stdio shim. A Python<->C parity gate ties
 * it to bcir/verify.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_VERIFY_H
#define BCIR_VERIFY_H

#include "bcir_cir.h"
#include "bcir_host_alloc.h"
#include "bcir_plan.h"
#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* R1-R8 + R12 + R14-R17 + R18 over the unit's claim graph + call graph. 1 == clean. */
int bcir_verify_unit(const bcir_unit *u, char *diag, size_t dn);

/* Hosted allocator-injected form used by the re-entrant C frontend. The
 * allocator is borrowed for the call; the verifier retains nothing. */
int bcir_verify_unit_with_allocator(const bcir_unit *u, char *diag, size_t dn,
                                    const bcir_host_allocator *allocator);

/* R9: a plan realizes every claim exactly once and its total cost is the sum of step costs. */
int bcir_verify_plan(const bcir_func *f, const bcir_plan *p, char *diag, size_t dn);

/* R10-R11: the hydrated StreamPack is well-formed (magic/version/CRC validate) and its segment
 * count matches the realizable (non-marker) claim count. */
int bcir_verify_pack(const uint8_t *pack, size_t len, uint32_t expect_segs, char *diag, size_t dn);

/* R13: a deterministic content digest of a function's claim graph (the provenance manifest). */
uint64_t bcir_provenance_digest(const bcir_func *f);

/* R21 (pointer-lifetime legality, §5.12): an ADVISORY walk over the optional `claim.lifetime` malloc/free
 * annotation, reporting each use-after-free / double-free via `report(funcname, kind, ctx)` (kind is
 * "use-after-free" or "double-free"). Separate from the verdict -- never affects bcir_verify_unit / r.ok. */
void bcir_verify_lifetime(const bcir_unit *u,
                          void (*report)(const char *funcname, const char *kind, void *ctx),
                          void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_VERIFY_H */
