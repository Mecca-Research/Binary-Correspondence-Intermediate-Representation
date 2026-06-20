/* Local aggregate initializers for a struct/union (#aggregate, the C twin of lower._agg_init): a
 * braced `struct cfg c = {.field=v, ...}` / `union u = {...}` lowers to a `= {0}` zero baseline (so
 * uninitialized members zero-fill, §6.7.10) + a c.store per initialized member. The driver config /
 * descriptor pattern. Both rails agree structurally and the emit is Clang-behaviour-equivalent.
 * (Local arrays + [i]= designators need a local array declarator -- a follow-on.) */
struct Cfg { unsigned baud; unsigned parity; unsigned stop; };
union Word { unsigned u; unsigned bits; };

unsigned config(unsigned x) { struct Cfg c = {.baud=x, .stop=x+1u}; return c.baud*1000u + c.parity*10u + c.stop; }
unsigned positional(unsigned x) { struct Cfg c = {x, x+2u, x+4u}; return c.baud + c.parity + c.stop; }
unsigned overlap(unsigned x) { union Word w = {.bits=x}; return w.u; }
