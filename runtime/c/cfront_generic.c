/* _Generic (#generic): C11 generic selection `_Generic(ctrl, T1: e1, ..., default: eN)` (§6.5.1.1) -- a
 * compile-time choice on the static type of the UNEVALUATED controlling expression. The first type-name
 * whose type matches (after lvalue/array decay; qualifiers ignored) wins, else `default`; only the chosen
 * association's expression is evaluated. Type identity: int / int32_t collapse (same width + signedness),
 * plain `char` is distinct from signed/unsigned char, `_Bool` is its own type, floats key on width, a
 * pointer on its pointee, an aggregate on its tag. Both rails read the controlling type the same way (the
 * twin off a speculatively-lowered value, then rolled back) and pick the same arm; the emit == Clang. */

/* a type tag: each matched association returns a distinct code (a wrong arm -> a wrong code) */
int g_int(int x)               { return _Generic(x, int: 10, long: 20, double: 30, default: 0); }
int g_long(long x)             { return _Generic(x, int: 10, long: 20, double: 30, default: 0); }
int g_uint(unsigned x)         { return _Generic(x, int: 1, unsigned: 2, long: 3, default: 0); }
int g_double(double x)         { return _Generic(x, float: 1, double: 2, int: 3, default: 0); }
int g_float(float x)           { return _Generic(x, float: 1, double: 2, int: 3, default: 0); }
int g_char(char x)             { return _Generic(x, char: 1, signed char: 2, unsigned char: 3, int: 4, default: 0); }
int g_ptr(int *p)              { return _Generic(p, int*: 1, char*: 2, long*: 3, default: 0); }
/* the controlling expression's type comes from the usual arithmetic conversions: int + long -> long */
int g_exprtype(int a, long b)  { return _Generic(a + b, int: 1, long: 2, double: 3, default: 0); }
/* no type-name matches -> the default association is selected */
int g_default(double x)        { return _Generic(x, int: 1, long: 2, default: 99); }
/* the value-yielding form: a different computation is selected per controlling type */
long g_compute(long x)         { return _Generic(x, int: x + 1, long: x * 2 - 3, default: x); }
