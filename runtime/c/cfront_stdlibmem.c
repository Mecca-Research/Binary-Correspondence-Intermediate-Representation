/* <stdlib.h> memory management (#stdlibmem) -- malloc / calloc / realloc / free as external libc edges,
 * emitted VERBATIM (opaque to R18, NOT bcir_-renamed), the seam the naked-pointer safety track (§5.12)
 * hangs lifetime annotations on. Previously the frontend treated them as undefined IN-UNIT calls
 * (`bcir_malloc(...)`, an R18-dirty undefined callee); now they lower clean -- the allocators return a
 * `void *` (assignable to any object pointer), `free` returns void. The fixtures allocate, fill, reduce,
 * and free -- the result is a deterministic VALUE (never an address), so the equivalence harness (each rail
 * mallocs independently, at different addresses) matches. */
#include <stdlib.h>

unsigned msum(unsigned n)
{
    unsigned m = n & 31u;
    unsigned *p = malloc(m * sizeof(unsigned));
    unsigned s = 0u;
    for (unsigned i = 0u; i < m; i++) { p[i] = i * 7u + 1u; s += p[i]; }
    free(p);
    return s;
}

unsigned czero(unsigned n)
{
    unsigned m = (n & 15u) + 1u;
    unsigned *p = calloc(m, sizeof(unsigned));      /* zero-initialized memory */
    unsigned s = p[0] + p[m - 1u];                  /* reads the zeros calloc guarantees */
    free(p);
    return s;
}

unsigned grow(unsigned n)
{
    unsigned m = (n & 7u) + 1u;
    unsigned *p = malloc(m * sizeof(unsigned));
    for (unsigned i = 0u; i < m; i++) p[i] = i + 1u;
    p = realloc(p, (m + 4u) * sizeof(unsigned));    /* grow -- preserves the first m elements */
    unsigned s = 0u;
    for (unsigned i = 0u; i < m; i++) s += p[i];
    free(p);
    return s;
}
