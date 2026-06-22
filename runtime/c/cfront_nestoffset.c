/* Nested member access at a NON-FIRST offset (#nestoffset): `t.q.a` where the enclosing member `q` is not
 * the struct's first member. The oracle's member-access base resolver dropped the enclosing member's byte
 * offset (it returned only a rid + type, no accumulated offset), so `t.p` and `t.q` aliased at offset 0 on
 * both reads AND writes; the twin was already correct. cfront_nestmember only ever nested through a first
 * member (offset 0), hiding it. Now the offset accumulates through each `.` hop -- a struct member, a
 * member array element, a deeper chain, and a non-first designated initializer (the read-back deferred by
 * #designate). Both rails agree and the emit == Clang. (Address-of a non-first nested member, `&t.q.a`,
 * needs twin `&member` support and the oracle offset folded into c.addrof -- a follow-on.) */

struct In   { int a, b; };
struct Two  { struct In p; struct In q; };          /* q is the 2nd member (offset 8) */
struct Arr  { int pad; int v[3]; };                 /* v is at offset 4 (non-first) */
struct Deep { int pad0; struct Two t; };            /* d.t.q.a -> pad0(4) + q(8) + a(0) = 12 */

/* read + write of a non-first nested struct member */
int no_rw(int x)     { struct Two t; t.p.a = x; t.p.b = x+1; t.q.a = x+2; t.q.b = x+3;
                       return t.p.a*1000 + t.p.b*100 + t.q.a*10 + t.q.b; }
/* a member array at a non-first offset */
int no_memarr(int x) { struct Arr s; s.pad = x; s.v[0] = x+1; s.v[1] = x+2; s.v[2] = x+3;
                       return s.v[0]*100 + s.v[1]*10 + s.v[2] + s.pad; }
/* a deeper chain reaching a non-first member through a non-first enclosing struct */
int no_deep(int x)   { struct Deep d; d.pad0 = x; d.t.q.a = x+5; d.t.p.b = x+6;
                       return d.t.q.a*100 + d.t.p.b*10 + d.pad0; }
/* a non-first designated initializer, now readable back (the #designate follow-on) */
int no_desig(int x)  { struct Two t = { .q.a = x, .q.b = x+1, .p.a = x+2 };
                       return t.q.a*100 + t.q.b*10 + t.p.a; }
