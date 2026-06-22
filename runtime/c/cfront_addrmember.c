/* Address-of a (nested) struct member (#addrmember): `&s.field`, `&t.q.a`, `&t.q` -- previously the twin
 * could not parse `&member` at all (only `&local`/`&param`/`&(compound literal)`), and the oracle emitted
 * the enclosing struct's address (right only for a first member at offset 0, wrong type/address otherwise).
 * Now `&member` resolves the member chain to a typed `(T *)((char *)&base + off)` -- the address used
 * through a pointer (read/write/compound-assign) and passed to a helper. Both rails agree and the emit ==
 * Clang. (Address-of a pointer/array member or a float member -- the field model lacks the float flag --
 * is a follow-on.) */

struct In  { int a, b; };
struct Two { struct In p; struct In q; };

/* a helper that mutates a value through a pointer -- &member as a call argument */
static void addone(int *p) { *p += 1; }

/* &member of the FIRST members (offset 0/4), used through pointers */
int am_first(int x)  { struct In s; s.a = x; s.b = x+1; int *pa = &s.a, *pb = &s.b; *pa += 10; *pb += 20;
                       return s.a*10 + s.b; }
/* &nested.member at a non-first offset (q at 8): &t.q.a (8), &t.q.b (12) */
int am_nested(int x) { struct Two t; t.p.a = x; t.q.a = x+1; t.q.b = x+2;
                       int *p = &t.q.a, *q = &t.q.b; *p += 5; *q += 6;
                       return t.p.a*100 + t.q.a*10 + t.q.b; }
/* &member that is itself a struct: &t.q is a `struct In *` (offset 8) */
int am_struct(int x) { struct Two t; t.q.a = x; t.q.b = x+1; struct In *ip = &t.q; ip->a += 3;
                       return ip->a*10 + ip->b; }
/* &member passed to a helper (the canonical use) */
int am_arg(int x)    { struct Two t; t.q.b = x; addone(&t.q.b); addone(&t.q.b); return t.q.b; }
