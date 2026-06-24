/* Native variable-length arrays (#vla): a 1-D stack VLA `T a[n]` whose size is a RUNTIME value. The size
 * is evaluated exactly once and snapshotted into a hidden immutable extent; the array is declared IN-BODY
 * (`T a[__ext];` -- a faithful stack allocation, no heap, no leak) and every `a[i]` is bounds-masked against
 * the snapshot (§5.12 recoverable extents -> `a[BCIR_CHK(rid, i, __ext, "fn:a")]`). Sizes are derived from
 * the input and all accesses are kept strictly in-bounds, so the two rails AND Clang agree for every input. */

unsigned vsum(unsigned n) {                 /* fill + reduce over a runtime-sized buffer */
    unsigned m = (n & 15u) + 1u;            /* m in [1, 16] */
    unsigned a[m];
    unsigned s = 0u;
    for (unsigned i = 0u; i < m; i++) { a[i] = i * 3u + n; s += a[i]; }
    return s;
}

int vsigned(int n) {                        /* a SIGNED-element VLA + signed arithmetic */
    int m = (n & 7) + 1;                    /* m in [1, 8] */
    int a[m];
    int s = 0;
    for (int i = 0; i < m; i++) { a[i] = i * 2 - n; s += a[i]; }
    return s;
}

unsigned vrev(unsigned n) {                 /* write forward, read reversed -- index arithmetic vs the extent */
    unsigned m = (n % 10u) + 1u;            /* m in [1, 10] */
    unsigned a[m];
    for (unsigned i = 0u; i < m; i++) a[i] = i + 1u;
    unsigned s = 0u;
    for (unsigned i = 0u; i < m; i++) s += a[m - 1u - i] * (i + 1u);
    return s;
}

unsigned vexpr(unsigned n) {                /* the size is an EXPRESSION (evaluated once at the declaration) */
    unsigned a[(n & 3u) + 2u];              /* size in [2, 5] */
    unsigned lim = (n & 3u) + 2u;
    unsigned s = 1u;
    for (unsigned i = 0u; i < lim; i++) { a[i] = (i ^ n) & 255u; s += a[i] * (i + 1u); }
    return s;
}
