/* continue -- jumps to the loop's continue point (the step for `for`, the bottom test for
 * do/while, the re-test for `while`), emitted as a per-loop `goto __cont_N`. Bounded loops. */
uint32_t odds(uint32_t seed)
{
    uint32_t acc = 0;
    uint32_t n = seed & 15;
    for (uint32_t i = 0; i < n; i = i + 1) {
        if ((i & 1) == 0) {
            continue;                 /* skip even -- the step (i = i+1) must still run */
        }
        acc = acc + i;
    }
    return acc;
}

uint32_t skip_small(uint32_t seed, uint32_t thresh)
{
    uint32_t acc = 0;
    uint32_t i = 0;
    uint32_t n = seed & 7;
    uint32_t t = thresh & 7;
    while (i < n) {
        i = i + 1;                    /* advance first so the loop is bounded */
        if (i < t) {
            continue;                 /* while-continue: re-test the cond */
        }
        acc = acc + i;
    }
    return acc;
}
