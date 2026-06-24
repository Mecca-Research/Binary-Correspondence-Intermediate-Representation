/* Regular local array declarations with brace initializers (#aoslocal): both the INFERRED-size form
 * `T a[] = {...}` (the outer count comes from the initializer -- max index + 1, gaps zero-fill) and the
 * EXPLICIT-size form `T a[N] = {...}`, for a SCALAR element AND a STRUCT element. An inferred `[]` must
 * size the storage from the initializer (an under-sized `a[0]` would let the per-element stores write past
 * it -- a #500-class silent miscompile); a struct-element array routes each `{...}` element through the
 * offset-based per-element struct store (`idx*sizeof(elem)`), riding the `= {0}` zero baseline. Both rails
 * lower an IDENTICAL claim sequence and stay Clang-behaviour-equivalent (a wrong size/offset is caught by
 * the `match` check). Indexed element-then-field `a[i].x` strides by the element struct. */

struct P { unsigned x, y; };

unsigned aoslocal_struct_inferred(unsigned i)
{
    struct P a[] = {{1u, 2u}, {3u, 4u}};          /* INFERRED count 2 -> `struct P a[2]` */
    return a[i & 1u].x;
}

unsigned aoslocal_struct_explicit(unsigned i)
{
    struct P b[2] = {{5u, 6u}, {7u, 8u}};         /* EXPLICIT size, struct element */
    return b[i & 1u].y;
}

unsigned aoslocal_scalar_inferred(unsigned i)
{
    unsigned c[] = {10u, 20u, 30u};               /* INFERRED count 3 -> `unsigned c[3]` */
    return c[i % 3u];
}

unsigned aoslocal_struct_partial(unsigned i)
{
    struct P d[2] = {{1u}};                        /* PARTIAL init -> the `= {0}` zero baseline fills the rest */
    return d[i & 1u].x + d[i & 1u].y;
}
