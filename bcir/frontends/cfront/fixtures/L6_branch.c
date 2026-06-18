#include <stdint.h>
/* L6: control flow — branches with early returns (-> compose.Cond). */
uint32_t l6_clamp(uint32_t x, uint32_t lo, uint32_t hi)
{
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}
