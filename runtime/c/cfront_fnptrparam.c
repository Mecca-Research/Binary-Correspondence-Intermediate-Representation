/* Direct (inline) function-pointer PARAMETERS `RET (*name)(PARAMS)` (#fnptrparam) -- the callback /
 * dispatch pattern WITHOUT a typedef alias (cfront_funcptr.c covers the typedef'd form). Each one
 * lowers + is called identically to the typedef'd shape: a `c.call.indirect` claim reading the pointer
 * then the actuals, an opaque external edge for R18. The difference is purely in EMIT -- there is no
 * source typedef to print, so the full inline signature is reconstructed (the C twin synthesizes a
 * `__bcir_fpN` prelude typedef; the Python oracle prints the `RET (*name)(PARAMS)` declarator directly),
 * and both compile standalone and behave identically under Clang. A direct call to the named `bias` in
 * the same function still travels the normal call graph (c.call:bias), so direct + indirect coexist. */

uint32_t bias(uint32_t v)
{
    return v + 7u;
}

uint32_t dispatch(int (*unary)(int), uint32_t (*binop)(uint32_t, uint32_t), int a, uint32_t b)
{
    int u = unary(a) + 1;                      /* single-argument inline funcptr param, indirect call */
    uint32_t w = binop((uint32_t)u, b);        /* two-argument inline funcptr param, indirect call */
    return bias(w) ^ b;                        /* a DIRECT call to the named bias still goes on the graph */
}
