/* §5.12 recoverable-extent SNAPSHOT (#extentsnap): a malloc/calloc whose element COUNT is a side-effect-free
 * EXPRESSION -- not a bare stable Name (the `malloc(m*sizeof)` fast path), but `(n&7)+1`, `a*b`, ... The
 * frontend evaluates the count once into a hidden IMMUTABLE local `__bcir_extK` at the allocation (immune to
 * any later mutation of its inputs) and bounds-checks every `p[i]` against that snapshot. Re-evaluating the
 * count for the snapshot is sound exactly because it is pure. Counts are kept small/bounded so every access
 * is in-bounds (the guard is transparent) and the result is a deterministic VALUE -- the equivalence harness
 * (each rail mallocs independently, at different addresses) matches. Both rails snapshot IDENTICALLY. */
#include <stdlib.h>

unsigned snap_sum(unsigned n)
{
    unsigned *p = malloc(((n & 7u) + 1u) * sizeof(unsigned));   /* count expr (n&7)+1, in 1..8 -> snapshot */
    unsigned c = (n & 7u) + 1u;
    unsigned s = 0u;
    for (unsigned k = 0u; k < c; k++) { p[k] = k * 3u + 1u; s += p[k]; }
    free(p);
    return s;
}

unsigned snap_calloc(unsigned n)
{
    unsigned a = (n & 3u) + 1u;
    unsigned *p = calloc(a * 2u, sizeof(unsigned));            /* count expr a*2, in 2..8 -> snapshot */
    unsigned s = p[0] + p[a * 2u - 1u];                        /* reads the zeros calloc guarantees */
    free(p);
    return s;
}
