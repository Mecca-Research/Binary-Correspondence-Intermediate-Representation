/* Function-pointer struct members, populated from NAMED functions (#fnptrmember): the canonical dispatch /
 * vtable / operations-struct pattern. A struct carries function-pointer members declared with the DIRECT
 * declarator `RET (*name)(params)` (no typedef needed); they are SET from named functions (`o->add = g_add`
 * -- function-to-pointer decay) and CALLED indirectly (`o->add(x,y)`).
 *
 * Two pieces beyond the prior typedef-member-call support: (1) the direct `(*name)(params)` member declarator
 * lays out an 8-byte (kind-3) field; (2) a bare function name used as a VALUE yields a funcptr (emitted as the
 * name), stored through a generic-funcptr lvalue `*(void(**)(void))slot = (void(*)(void))g` -- a plain
 * `memcpy(&g,8)` would copy the function's CODE, since `&g` IS the function's address. Function pointers
 * round-trip through the cast, and the call reads the member's real type. Both rails emit identically and
 * match Clang on every target. (The members return `unsigned` like the indirect-call value model -- the
 * `c.call.imember` result is uint-typed on both rails; a signed funcptr return is a follow-on.) */

struct Ops {
    unsigned  tag;
    unsigned  (*add)(unsigned, unsigned);     /* direct-declarator funcptr members */
    unsigned  (*mul)(unsigned, unsigned);
    unsigned  (*inc)(unsigned);
    unsigned  total;
};

static unsigned op_add(unsigned a, unsigned b) { return a + b; }
static unsigned op_mul(unsigned a, unsigned b) { return a * b; }
static unsigned op_inc(unsigned a)             { return a + 1u; }

unsigned dispatch(struct Ops *o, unsigned x, unsigned y)
{
    o->tag   = x ^ y;
    o->add   = op_add;              /* set each member from a named function */
    o->mul   = op_mul;
    o->inc   = op_inc;
    o->total = o->add(x, y) + o->mul(x, y) + o->inc(x);   /* call through the members */
    return o->tag + o->total + o->add(o->tag, y) + o->inc(o->mul(x, y));
}
