/* Pointer-arithmetic dereference (L3): `*p` loads through a pointer, and `*(p + i)` is the
 * pointer-arithmetic spelling of `p[i]` -- both lower to a load on the existing index/load machinery
 * (`*p` a one-read deref load, `*(p + i)` the two-read `p[i]` form), Clang-behaviour-equivalent. */
uint32_t deref(uint32_t *p, uint32_t i)
{
    uint32_t a = *p;
    uint32_t b = *(p + i);
    return a + b;
}
