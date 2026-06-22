/* _Atomic local objects (#atomiclocal): `_Atomic int a;` (the qualifier form) and `_Atomic(int) a;` (the
 * atomic type-specifier form), plus `const _Atomic` and a pointer-to-atomic `_Atomic(int) *p`. The global
 * `_Atomic int g;` already worked; a LOCAL was rejected -- the statement-level declaration detector
 * (twin `looks_decl` / oracle `_is_decl_start`) did not list `_Atomic`, and the `_Atomic(T)` paren spelling
 * was unparsed. An _Atomic local with automatic storage is not shared, so its single-threaded semantics
 * equal the plain underlying type; both rails lower the arithmetic identically and the emit == Clang. */

int  a_qual(int x)  { _Atomic int a = x; a += 3; a *= 2; a -= 1; return a; }
int  a_paren(int x) { _Atomic(int) a = x * 2; a |= 1; a ^= 5; return a; }
long a_long(long x) { _Atomic long a = x; a <<= 1; a += 100; return a; }
int  a_const(int x) { const _Atomic int a = x * 3 + 7; return a + 1; }
int  a_ptr(int x)   { int v = x; _Atomic(int) *p = &v; *p += 4; return v + *p; }
