/* Function pointers (HAL dispatch): a typedef'd function-pointer type passed as a parameter and
 * called indirectly -- the canonical embedded driver-op / dispatch-table pattern. The indirect call
 * lowers to a `c.call.indirect` claim (reads: the function pointer, then the actuals); R18 leaves it
 * an opaque external edge (no named callee, so no recursion / callee-resolution constraint applies),
 * while the *direct* call to `bias` in the same function still travels the normal call graph. */
typedef uint32_t (*binop_fn)(uint32_t, uint32_t);

uint32_t bias(uint32_t v)
{
    return v + 7u;
}

uint32_t apply(binop_fn op, uint32_t a, uint32_t b)
{
    return bias(op(a, b));
}
