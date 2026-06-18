/* Function-pointer struct members (HAL dispatch table): a struct of function pointers, called
 * indirectly through `o->fn(args)`. The member access + call fuse into one `c.call.imember:<field>`
 * claim (reads: the struct base, then the actuals) emitted verbatim as `o->fn(args)` -- so no 8-byte
 * function-pointer value has to ride in the 4-byte value model. R18 leaves it an opaque external
 * edge. The canonical driver-vtable / operations-struct pattern. */
typedef uint32_t (*binop_fn)(uint32_t, uint32_t);

struct ops { binop_fn add2; binop_fn mul2; };

uint32_t run(struct ops *o, uint32_t a, uint32_t b)
{
    return o->add2(a, b) + o->mul2(a, b);
}
