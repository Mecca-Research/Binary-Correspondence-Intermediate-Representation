/*===- bcir_hydrate.h - claim graph + plan -> StreamPack bytes -------------===
 *
 * The C twin of bcir/gem/hydrate: serializes a planned claim graph (bcir_cir.h +
 * bcir_plan.h) into the frozen StreamPack binary ABI (bcir_streampack.h) -- one segment
 * per claim, in claim order. The bytes it writes are accepted by the bounds-checked
 * decoder (bcir_runtime.c) and run by the executor (bcir_exec.c), so this closes the
 * loop with no Python:
 *
 *     C source -> bcir_cfront -> claim graph -> bcir_plan -> bcir_hydrate -> bcir_exec
 *
 * Freestanding (no libc): a driver builds + runs its compiled artifact in place. Bounds-
 * checked: never writes past `cap` (returns BCIR_ERR_NOSPACE).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_HYDRATE_H
#define BCIR_HYDRATE_H

#include "bcir_cir.h"
#include "bcir_plan.h"
#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Serialize `f` (planned by `plan`) into a StreamPack in `buf[0..cap)`; writes the
 * encoded length to *out_len. Returns BCIR_ERR_NOSPACE if `cap` is too small. */
BCIR_NODISCARD bcir_status bcir_hydrate(const bcir_func *f, const bcir_plan *plan,
                                        uint8_t *buf, size_t cap, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_HYDRATE_H */
