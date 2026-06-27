/* C23 function attributes (#unsequenced / A1.3): the value-neutral hints `[[unsequenced]]` and
 * `[[reproducible]]` on a FUNCTION definition. They are advisory (no effect on computed values), so
 * both rails parse + CONSUME the attribute and DROP it from the emitted C -- dropping a value-neutral
 * hint keeps behaviour byte-identical under Clang. Recorded as a fusion-legality signal surfaced in the
 * dual-rail `repro=` summary (here 2: us_scale + us_combine carry the hint; us_plain does not). */
[[unsequenced]] uint32_t us_scale(uint32_t a, uint32_t b)
{
    uint32_t s = a * 3u + b;
    return s ^ (a & b);
}
[[reproducible]] uint32_t us_combine(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t t = (a + b) * (c | 1u);
    return t - (a ^ c);
}
uint32_t us_plain(uint32_t a, uint32_t b)        /* NO attribute -> does not count toward repro= */
{
    return (a << 1) + (b >> 1);
}
