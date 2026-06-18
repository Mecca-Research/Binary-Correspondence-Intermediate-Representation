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
#include "bcir_plan.h"
#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* R1-R8 + R12 + R14-R17 + R18 over the unit's claim graph + call graph. 1 == clean. */
int bcir_verify_unit(const bcir_unit *u, char *diag, size_t dn);

/* R9: a plan realizes every claim exactly once and its total cost is the sum of step costs. */
int bcir_verify_plan(const bcir_func *f, const bcir_plan *p, char *diag, size_t dn);

/* R10-R11: the hydrated StreamPack is well-formed (magic/version/CRC validate) and its segment
 * count matches the realizable (non-marker) claim count. */
int bcir_verify_pack(const uint8_t *pack, size_t len, uint32_t expect_segs, char *diag, size_t dn);

/* R13: a deterministic content digest of a function's claim graph (the provenance manifest). */
uint64_t bcir_provenance_digest(const bcir_func *f);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_VERIFY_H */
