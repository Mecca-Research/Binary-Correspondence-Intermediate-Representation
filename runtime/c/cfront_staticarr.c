/* A `static` local array or aggregate keeps its shape: a scalar array, a two-dimensional one, an array of
 * pointers indexed twice, a one-element array, a volatile array, a struct and an array of structs. Both
 * rails lowered each as the object it is -- its subscripts guarded against its extent -- and declared it as
 * a scalar (`static uint32_t hist = 0u;`), so the emitted C did not compile. The original and the emitted
 * twin each keep their own statics and see the same inputs, so the accumulated state agrees call for call. */
#include <stdint.h>

struct Acc {
    uint32_t n;
    uint16_t last;
};

uint32_t static_arrays(uint16_t *a, uint32_t i)
{
    static uint32_t hist[3];
    static uint16_t grid[2][4];
    static uint16_t *rows[2];
    static uint32_t one[1];
    static int8_t sgn[4];
    static volatile uint16_t va[2];
    static struct Acc acc;
    static struct Acc accs[2];
    uint32_t k = i % 3u, r = i & 1u, c = (i >> 1) & 3u;
    hist[k] += i;
    grid[r][c] = (uint16_t)(grid[r][c] + a[c]);
    rows[0] = a + 4;
    rows[1] = a;
    one[0] ^= i;
    sgn[c] = (int8_t)(sgn[c] - (int8_t)i);
    va[r] = (uint16_t)(va[r] + i);
    acc.n += i;
    acc.last = (uint16_t)i;
    accs[r].n += i + 1u;
    return hist[0] + 3u * hist[1] + 5u * hist[2] + grid[r][c] + rows[r][c] + one[0] + (uint32_t)(int32_t)sgn[c]
           + va[0] + va[1] + acc.n + acc.last + accs[0].n + 7u * accs[1].n;
}
