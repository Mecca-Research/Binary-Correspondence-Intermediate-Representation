#include <stdint.h>
/* L2: struct layout + member access. */
struct point { uint32_t x; uint32_t y; };
uint32_t l2_sumsq(struct point p)
{
    uint32_t a = p.x * p.x;
    uint32_t b = p.y * p.y;
    return a + b;
}
