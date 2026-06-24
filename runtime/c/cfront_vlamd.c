/* Multi-dimensional stack VLAs (#vlamd): `T a[m][n]` (and 3-D) where one or more dims are runtime. Each dim
 * is snapshotted once (canonical order: dims 0..k-1, then the product), the array is declared IN-BODY as a
 * FLAT `T a[__ext_total];` (the same row-major memory layout as `T a[m][n]`), and the Horner-flattened index
 * `i*n + j` (the inner-dim runtime stride) is bounds-masked against the total -- a[BCIR_CHK(rid, i*n+j, m*n)].
 * Dims are derived from the inputs and kept tiny so the allocation is small and every access is in-bounds for
 * ANY input, so the two rails AND Clang agree. */

unsigned md2(unsigned p, unsigned q) {              /* a 2-D VLA fill + reduce, row-major */
    unsigned m = (p & 3u) + 1u;                     /* m in [1, 4] */
    unsigned n = (q & 3u) + 1u;                     /* n in [1, 4] */
    unsigned a[m][n];
    unsigned s = 0u;
    for (unsigned i = 0u; i < m; i++)
        for (unsigned j = 0u; j < n; j++) { a[i][j] = i * n + j + p; s += a[i][j]; }
    return s;
}

int md2s(int p, int q) {                            /* a signed-element 2-D VLA */
    int m = (p & 3) + 1;
    int n = (q & 3) + 2;                            /* n in [2, 5] */
    int a[m][n];
    int s = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) { a[i][j] = i * n - j + p; s += a[i][j]; }
    return s;
}

unsigned md3(unsigned p) {                          /* a 3-D VLA -- l*m*n flatten, double Horner */
    unsigned l = (p & 1u) + 1u;                     /* l in [1, 2] */
    unsigned m = (p & 1u) + 2u;                     /* m in [2, 3] */
    unsigned n = (p & 3u) + 1u;                     /* n in [1, 4] */
    unsigned a[l][m][n];
    unsigned s = 0u;
    for (unsigned i = 0u; i < l; i++)
        for (unsigned j = 0u; j < m; j++)
            for (unsigned k = 0u; k < n; k++) { a[i][j][k] = (i + j + k + p) & 255u; s += a[i][j][k]; }
    return s;
}

unsigned mdmix(unsigned p) {                        /* a runtime-by-literal dim `a[m][4]` */
    unsigned m = (p % 5u) + 1u;                     /* m in [1, 5] */
    unsigned a[m][4];
    unsigned s = 1u;
    for (unsigned i = 0u; i < m; i++)
        for (unsigned j = 0u; j < 4u; j++) { a[i][j] = (i * 4u + j) ^ p; s += a[i][j] * (j + 1u); }
    return s;
}
