/* typeof (#typeof): C23 `typeof(type-name)` / `typeof(variable)` (and the GNU `__typeof__` /
 * C23 `typeof_unqual` spellings) as a type-specifier, resolving to the operand's type. Both rails
 * resolve a type-name operand (including `typeof(int*)`), a bare in-scope variable `typeof(x)`, and a
 * general EXPRESSION operand `typeof(a+b)` -- the oracle infers the operand's static type without
 * evaluating it, the twin speculatively lowers it then rolls the emission back (the operand is
 * unevaluated). The resolved type must MATTER -- each case is built so the WRONG type (e.g. int instead
 * of long, or signed instead of unsigned, or a missing short truncation) would diverge -- and is driven
 * by a value differential == Clang (which compiles typeof natively) on both rails. */

struct TS  { int x, y; };
struct TS2 { int x; long y; };

int      to_width(long a)    { typeof(a) acc = a * 1000000L + a; return (int)(acc >> 20); } /* long math, not int */
unsigned to_sign(unsigned a) { typeof(a) m = a - 5u; return m >> 1; }            /* a logical (unsigned) shift */
int      to_typename(int a)  { typeof(short) s = (short)(a * 1000); return s + 1; } /* a short truncation/wrap */
int      to_ptr(int a)       { int v = a; typeof(int *) p = &v; return *p * 3; }  /* a `typeof(int*)` deref */
int      to_struct(int a)    { struct TS s = {a, a + 1}; typeof(s) t = s; return t.x * 10 + t.y; } /* struct copy */
long     to_unqual(long a)   { __typeof__(a) b = a + 7; return b * 2; }            /* the GNU spelling -> long */

/* --- typeof of a general expression operand --------------------------------------------------------- */
/* typeof(a + b): a is long, b is int -> long (usual arithmetic conversions). A wrong `int` would
 * truncate the wide product before the shift. */
int to_ebinop(long a)     { int b = 3; typeof(a + b) x = a * 1000000L + b; return (int)(x >> 20); }
/* typeof(a - 1u): a is unsigned, 1u unsigned -> unsigned int. A wrong signed type makes `>>` an
 * arithmetic (sign-propagating) shift -- diverges once the high bit is set. */
unsigned to_ebinsign(unsigned a) { typeof(a - 1u) m = a - 5u; return m >> 1; }
/* typeof((short)e): the cast expression's type is `short`. A wrong `int` loses the 16-bit wrap. */
int to_ecast(int a)       { typeof((short)a) s = (short)(a * 1000); return s + 1; }
/* typeof(*p): dereference of `long *` -> long. A wrong `int` truncates the wide load. */
int to_ederef(long a)     { long v = a * 1000000L; long *p = &v; typeof(*p) y = *p + a; return (int)(y >> 20); }
/* typeof(s.y): a `long` struct member -> long. A wrong `int` truncates the wide field value. */
int to_emember(int a)     { struct TS2 s; s.x = a; s.y = (long)a * 1000000L + a; typeof(s.y) w = s.y; return (int)(w >> 20); }
/* typeof(arr[1]): the element type of a `long` array -> long. A wrong `int` truncates the element. */
int to_eindex(long a)     { long arr[4]; arr[1] = a * 1000000L + a; typeof(arr[1]) e = arr[1]; return (int)(e >> 20); }
