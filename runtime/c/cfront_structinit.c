/* Struct-valued member set (#structinit): a struct/union-typed member assigned a whole struct VALUE --
 * either in an aggregate initializer (`struct Outer o = { inner, ... }` / `{ (struct Pt){...}, ... }`)
 * or by a direct member assignment (`o.p = q`). The member store copies the whole object (a memcpy of
 * the member's size), not a scalar `uintN _v = <struct>` -- which is a type error in C and would also
 * under-read a member wider than the scalar temp. Both rails agree structurally and the emit is
 * Clang-behaviour-equivalent (the twin's emit previously failed to compile for this case). */

struct Pt    { int x, y; };
struct Outer { struct Pt p; int z; };
struct Wide  { int tag; long val; };
struct W2    { struct Wide w; int k; };

/* a struct-typed member initialized by a struct VARIABLE */
int si_var(int a, int b)    { struct Pt inner = {a, b}; struct Outer o = {inner, a + b};
                              return o.p.x * 100 + o.p.y * 10 + o.z; }
/* a struct-typed member initialized by a struct COMPOUND LITERAL */
int si_lit(int a, int b)    { struct Outer o = {(struct Pt){a, b}, a - b};
                              return o.p.x * 100 + o.p.y * 10 + o.z; }
/* a nested member carrying a wide `long` -- the object copy must move all 8 bytes of the inner struct */
int si_wide(int a, int b)   { struct Wide w = {a, (long)b * 1000000L + a}; struct W2 s = {w, a ^ b};
                              return (int)((s.w.val >> 20) + s.w.tag * 7 + s.k); }
/* a direct member assignment from a struct value: `o.p = q` */
int si_assign(int a, int b) { struct Outer o = {(struct Pt){0, 0}, 0}; struct Pt q = {a, b};
                              o.p = q; o.z = a + b; return o.p.x * 100 + o.p.y * 10 + o.z; }
