#include <stdint.h>
/* L1: fixed-width integer expressions. */
uint32_t l1_compute(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t t = a + b * c;
    uint32_t u = (t ^ 0x5A) << 2;
    return u - a;
}
