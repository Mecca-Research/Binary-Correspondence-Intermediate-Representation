/* ternary `?:` -- a scalar select (clamp + a nested, right-associative pick). Lowered by both
 * rails to a c.select claim; the emitter renders the real C `(cond ? a : b)`. */
uint32_t clamp3(uint32_t x, uint32_t lo, uint32_t hi)
{
    uint32_t a = x < lo ? lo : x;
    uint32_t b = a > hi ? hi : a;
    uint32_t nested = x == 0u ? lo : (x > hi ? hi : x);
    return b + nested;
}
