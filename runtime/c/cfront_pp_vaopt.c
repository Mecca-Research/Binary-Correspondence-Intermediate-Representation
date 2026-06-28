/* L7 adversarial preprocessor corpus -- variadic macros and C23 `__VA_OPT__` (C 6.10.4.1).
 * `__VA_ARGS__` is the (possibly empty) trailing arguments; `__VA_OPT__(content)` expands to `content`
 * only when `__VA_ARGS__` is non-empty -- the standard comma-elision idiom for a trailing-comma logger.
 * Empty vs non-empty cases are exercised, plus a stringize of `__VA_ARGS__`. Single-valued: clang/gcc
 * agree exactly (token-for-token; only inter-token spacing differs, which the differential normalizes). */
#define LOG(fmt, ...) log_impl(fmt __VA_OPT__(,) __VA_ARGS__)
#define COUNT(...) (0 __VA_OPT__(+ 1))
#define VSTR(...) #__VA_ARGS__
LOG("no args")
LOG("with", 1, 2, 3)
COUNT()
COUNT(a)
COUNT(a, b)
VSTR()
VSTR(x,y,z)
