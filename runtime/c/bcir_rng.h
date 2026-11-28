/*===- bcir_rng.h - a deterministic counter-based random-number generator --------===
 *
 * The C subset audit's first native addition (docs/research/BCIR_C_SUBSET_AUDIT_2026-11-10.md): Philox4x32-10, a
 * key plus a counter and no hidden state, so element i of a stream is the same number whatever thread, batch or
 * order asks for it -- the property the trainer already holds for its arithmetic. Freestanding: no allocation, no
 * libc, every function total. The Python twin is bcir/frontends/models/rng.py; both are held to the generator's
 * published known-answer vectors (Random123, D. E. Shaw Research: J. K. Salmon, M. A. Moraes, R. O. Dror and
 * D. E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", SC11) and to each other.
 *
 * A stream is a key (two words) and a 64-bit stream id. Element `index` of it is lane `index & 3` of the Philox
 * block whose counter is {lo32(index >> 2), hi32(index >> 2), lo32(stream), hi32(stream)}.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_RNG_H
#define BCIR_RNG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One Philox4x32-10 block: `out` = the bijection of `counter` under `key`. */
void bcir_philox4x32_10(const uint32_t counter[4], const uint32_t key[2], uint32_t out[4]);

/* Element `index` of the stream (`key`, `stream`). */
uint32_t bcir_rng_u32(const uint32_t key[2], uint64_t stream, uint64_t index);

/* Elements [first, first + n) of the stream into `out`; returns first + n, the next element's index. */
uint64_t bcir_rng_fill_u32(const uint32_t key[2], uint64_t stream, uint64_t first, uint32_t *out, size_t n);

/* A float in [0, 1) from one element: its top 24 bits times 2^-24, exact. */
float bcir_rng_unit_f32(uint32_t bits);

/* A double in [0, 1) from two elements, `hi` the earlier: the top 53 of their 64 bits times 2^-53, exact. */
double bcir_rng_unit_f64(uint32_t hi, uint32_t lo);

/* A uniform integer in [0, bound), `bound` >= 1, by multiply-and-shift with rejection (D. Lemire, "Fast Random
 * Integer Generation in an Interval", ACM TOMACS 29(1), 2019): it consumes elements from `*index` on, one unless a
 * draw falls in the rejected band, and advances `*index` past those it took. A bound of 0 gives 0 and takes none. */
uint32_t bcir_rng_below(const uint32_t key[2], uint64_t stream, uint64_t *index, uint32_t bound);

/* A permutation of [0, n) into `out` by Fisher-Yates from the stream's element `*index` on (j = below(i + 1) for i
 * from n - 1 down to 1, then swap out[i] and out[j]); `*index` advances past what it took. */
void bcir_rng_permutation(const uint32_t key[2], uint64_t stream, uint64_t *index, uint32_t *out, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_RNG_H */
