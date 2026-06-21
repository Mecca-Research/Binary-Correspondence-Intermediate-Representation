/* typeof (#typeof): C23 `typeof(type-name)` / `typeof(variable)` (and the GNU `__typeof__` /
 * C23 `typeof_unqual` spellings) as a type-specifier, resolving to the operand's type. Both rails
 * resolve a type-name operand (including `typeof(int*)`) and a bare in-scope variable `typeof(x)`; a
 * general expression operand (`typeof(a+b)`) is a deferred follow-on. The resolved type must MATTER --
 * each case is built so the WRONG type (e.g. int instead of long, or signed instead of unsigned) would
 * diverge -- and is driven by a value differential == Clang (which compiles typeof natively) on both
 * rails. */

struct TS { int x, y; };

int      to_width(long a)    { typeof(a) acc = a * 1000000L + a; return (int)(acc >> 20); } /* long math, not int */
unsigned to_sign(unsigned a) { typeof(a) m = a - 5u; return m >> 1; }            /* a logical (unsigned) shift */
int      to_typename(int a)  { typeof(short) s = (short)(a * 1000); return s + 1; } /* a short truncation/wrap */
int      to_ptr(int a)       { int v = a; typeof(int *) p = &v; return *p * 3; }  /* a `typeof(int*)` deref */
int      to_struct(int a)    { struct TS s = {a, a + 1}; typeof(s) t = s; return t.x * 10 + t.y; } /* struct copy */
long     to_unqual(long a)   { __typeof__(a) b = a + 7; return b * 2; }            /* the GNU spelling -> long */
