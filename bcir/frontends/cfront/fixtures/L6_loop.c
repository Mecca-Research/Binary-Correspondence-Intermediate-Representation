#include <stdint.h>
/* L6: control flow — a bounded while loop with mutable accumulators. */
uint32_t l6_weighted_sum(uint32_t seed, uint32_t step)
{
    uint32_t sum = 0;
    uint32_t i = 0;
    uint32_t n = seed & 15;        /* bound the loop: 0..15 iterations */
    while (i < n) {
        sum = sum + step * i;
        i = i + 1;
    }
    return sum;
}
