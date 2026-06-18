/* for loops -- desugared onto the existing while machinery: init; while(cond){ body; step }.
 * Loop counts are masked small (the equivalence harness seeds full-range scalars). */
uint32_t for_sum(uint32_t seed, uint32_t step)
{
    uint32_t acc = 0;
    uint32_t n = seed & 15;
    for (uint32_t i = 0; i < n; i = i + 1) {
        acc = acc + step * i;
    }
    return acc;
}

uint32_t for_poly(uint32_t x, uint32_t k)
{
    uint32_t r = 1;
    uint32_t m = k & 7;
    for (uint32_t j = 1; j <= m; j = j + 1) {
        r = r * x + j;
    }
    return r;
}
