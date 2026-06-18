/* do/while + break -- the cond is tested at the bottom (body runs at least once); break exits the
 * loop early. Loop counts are bounded (the equivalence harness seeds full-range scalars). */
uint32_t dw_sum(uint32_t seed)
{
    uint32_t i = seed & 7;
    uint32_t acc = 0;
    do {
        acc = acc + i;
        if (acc > 20) {
            break;
        }
        i = i + 1;
    } while (i < 10);
    return acc;
}

uint32_t dw_count(uint32_t x, uint32_t cap)
{
    uint32_t c = 0;
    uint32_t v = x & 31;
    uint32_t lim = cap & 7;
    do {
        c = c + 1;
        v = v + 3u;
        if (c >= lim) {
            break;
        }
    } while (v < 50u);
    return c;
}
