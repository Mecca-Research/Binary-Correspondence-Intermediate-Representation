/* Nested / chained designated initializers (#designate): a designator LIST `.a.b`, `.v[i]`, `.m[i][j]`
 * (and deeper) in a braced aggregate initializer, beyond the single `.field=` / `[i]=` the frontend
 * already handled. The chain resolves to a cumulative byte offset within the object -- `.field` descends a
 * (nested value-)struct/union member, `[i]` folds a constant index into a member array (Horner over its
 * dims) -- and the value stores there (gaps zero-fill, §6.7.10). The config-table idiom. Both rails walk
 * the layout identically and the emit is Clang-behaviour-equivalent. (Read-back here uses only first-
 * member nesting; a non-first nested member -- `t.q.a` with q not first -- needs the member-access offset
 * fix, a follow-on.) */

struct In { int a, b; };
struct Out { struct In in; int z; };
struct Sa { int v[3]; };
struct Md { int m[2][3]; };
struct L1 { int x; };
struct L2 { struct L1 l1; int y; };
struct L3 { struct L2 l2; int w; };

/* a 2-level struct chain (.in.a) mixed with a top-level member (.z) */
int desig_chain(int x)  { struct Out o = { .in.a = x, .in.b = x + 1, .z = x + 2 }; return o.in.a*100 + o.in.b*10 + o.z; }
/* a struct member array, designators out of order */
int desig_memarr(int x) { struct Sa s = { .v[2] = x, .v[0] = x + 1, .v[1] = x + 2 }; return s.v[0]*100 + s.v[1]*10 + s.v[2]; }
/* a 2-D struct member array */
int desig_md(int x)     { struct Md s = { .m[1][2] = x, .m[0][1] = x + 5, .m[1][0] = x - 3 }; return s.m[1][2]*100 + s.m[0][1]*10 + s.m[1][0]; }
/* an out-of-order mix of a top member and a nested one */
int desig_mix(int x)    { struct Out o = { .z = x, .in.b = x * 2 }; return o.z*10 + o.in.b; }
/* a 3-level struct chain (.l2.l1.x) */
int desig_deep(int x)   { struct L3 t = { .l2.l1.x = x, .l2.y = x + 1, .w = x + 2 }; return t.l2.l1.x*100 + t.l2.y*10 + t.w; }
