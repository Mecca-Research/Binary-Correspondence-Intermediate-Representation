/* Logical operators && / || (short-circuit; the condition idiom) and unary plus (a no-op). The C
 * rail previously parsed only the bitwise & / | -- a user-written `a && b` did not parse (the switch
 * desugar built `||` internally). Both lower to a `c.bin.land` / `c.bin.lor` claim and emit `&&` /
 * `||` verbatim (Clang short-circuits; equivalent for the side-effect-free subset). */
uint32_t logic(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t f = (a > 0u) && (b < c);
    uint32_t g = (a == b) || (c != 0u);
    return +f + g;
}
