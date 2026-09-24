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

/* Serialize `f` (optionally planned by `plan`) into a StreamPack in `buf[0..cap)`;
 * writes the encoded length to *out_len. A non-NULL plan must cover the function exactly.
 * All graph/size/capacity checks occur before `buf` is touched; on failure *out_len is zero
 * and no partial artifact is written. Returns BCIR_ERR_NOSPACE if `cap` is too small. */
BCIR_NODISCARD bcir_status bcir_hydrate(const bcir_func *f, const bcir_plan *plan,
                                        uint8_t *buf, size_t cap, size_t *out_len);

/* The per-step freeze of the dynamic-graph builder (G16, S3-C): the same segments and trace
 * records, as a StreamPack v4 bound to the live registry -- the vector `gens[0..n_gens)` (RIDs
 * strictly ascending, at least one: a frozen step binds to a registry; BCIR_ERR_GENERATION
 * otherwise) appended, its maxima as the header map_gen/data_gen, `topo_gen` the registry's, and
 * every segment carrying the v3 defaults (dispatch core, channel "host"). On top of
 * bcir_hydrate's laws, claim ids must ascend strictly in claim order and every RID a realizable
 * claim reads or writes must be declared by the vector (BCIR_ERR_PROVENANCE): a step that touches
 * a resource its registry does not carry is refused at the freeze, not at execution. The same
 * preflight-then-write discipline: on failure *out_len is zero and nothing is written. The Python
 * oracle is bcir/gem/handoff.py::freeze_claims (byte-identical). */
BCIR_NODISCARD bcir_status bcir_hydrate_generations(const bcir_func *f, const bcir_plan *plan,
                                                    uint32_t topo_gen,
                                                    const bcir_generation_view *gens,
                                                    size_t n_gens, uint8_t *buf, size_t cap,
                                                    size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_HYDRATE_H */
