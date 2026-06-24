/* Runtime `sizeof` of a 1-D stack VLA (#vlasizeof): `sizeof a` where `a` is a VLA is a RUNTIME value --
 * the snapshot extent times the element size, emitted as `(size_t)((size_t)__bcir_extK * sizeof(elem))` on
 * BOTH rails. `sizeof a[0]` (an element) stays the STATIC element size. Every loop bound is the runtime
 * extent (or `sizeof a / sizeof a[0]`, which recovers the count), so accesses stay in-bounds for ALL inputs,
 * and the result folds in the byte size so a stale static fold (0 / a fixed constant) diverges from Clang. */

unsigned vsz(unsigned n) {
    unsigned m = (n & 7u) + 1u;                 /* m in [1, 8] */
    unsigned a[m];
    unsigned bytes = (unsigned)sizeof a;        /* RUNTIME: m * sizeof(unsigned) == m*4 */
    unsigned elt   = (unsigned)sizeof a[0];     /* STATIC: sizeof(unsigned) == 4 */
    unsigned cnt   = bytes / elt;               /* recovers the count == m */
    unsigned s = 0u;
    for (unsigned i = 0u; i < cnt; i++) { a[i] = i + n; s += a[i]; }
    return s + bytes + elt;
}

int vszs(int n) {                               /* signed-element VLA, parenthesized `sizeof(a)` */
    int m = (n & 3) + 1;                        /* m in [1, 4] */
    int a[m];
    int bytes = (int)sizeof(a);                 /* RUNTIME: m * sizeof(int) */
    int s = 0;
    for (int i = 0; i < m; i++) { a[i] = i - n; s += a[i]; }
    return s + bytes;
}

unsigned vszmix(unsigned n) {                   /* sizeof feeding control flow + arithmetic */
    unsigned m = (n % 6u) + 2u;                 /* m in [2, 7] */
    unsigned a[m];
    unsigned total = (unsigned)sizeof a;        /* RUNTIME */
    unsigned half = (total / (unsigned)sizeof a[0]) / 2u + 1u;   /* <= m, in-bounds */
    unsigned s = 7u;
    for (unsigned i = 0u; i < half; i++) { a[i] = (i ^ n) & 15u; s += a[i] * (i + 1u); }
    return s + total;
}
