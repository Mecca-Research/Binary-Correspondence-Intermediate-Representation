/* A subscript chain through pointer elements: `q[j][i]` on an array of pointers, `pp[j][i]` on a pointer
 * to pointers, `pps[0][j][i]` through an array of those, `*(q[j] + i)`, and a parameter declared
 * `T *rows[]` index what the element HOLDS -- the element is loaded, then indexed. Both rails once
 * flattened the chain into the base as if it were a two-dimensional array, reading `q[j + i]` (a pointer,
 * returned as a number) and `pp[j + i]` (typed as the pointee); the twin also typed `pp[j]` as a value,
 * declared `T *rows[]` one level short and `T *a[1]` as a scalar. A true two-dimensional array still
 * flattens. Every store is a function of the arguments alone, so the original and the emitted twin leave
 * the same bytes. */
static uint32_t pick(uint16_t *rows[], uint32_t j, uint32_t k)   /* an array of pointers as a parameter
                                                                  * (`argv`): `uint16_t **`, one level more */
{
    return rows[j & 1u][k & 3u] + *rows[j & 1u];
}

/* The table `pick` reads: file-scope, so no automatic array is lent to a call (every local array of the
 * corpus stays proved private, `escape.unproved`). Both runs store the same two pointers. */
static uint16_t *table[2];

uint32_t ptr_index(uint16_t *a, int8_t *b, uint32_t i, uint32_t j)
{
    uint16_t *q[2];
    q[0] = a;
    q[1] = a + 8;
    uint16_t **pp = q;
    int8_t *r[2];
    r[0] = b;
    r[1] = b + 4;
    uint16_t **pps[1];
    pps[0] = pp;
    uint16_t m[2][4];
    table[0] = a + 4;
    table[1] = a;
    uint32_t k = i & 3u, s = j & 1u;
    m[s][k] = (uint16_t)(i + j);
    q[s][k] = (uint16_t)(i * 3u + j);
    pp[s ^ 1u][k] = (uint16_t)(j << 2);
    q[s ^ 1u][(k + 1u) & 3u] |= (uint16_t)0x40u;
    pps[0][s ^ 1u][3] = (uint16_t)(i >> 1);
    uint16_t *e = pp[s];
    int32_t neg = r[s][k] >> 1;
    return (uint32_t)(e[k] + pp[s ^ 1u][k] + q[0][3] + pps[0][s][k] + m[s][k] + *(q[s] + 2)) + (uint32_t)neg
           + pick(table, i, j);
}
