/* L7 adversarial preprocessor corpus -- one rescan, left to right, with hide sets (C11 6.10.3.4, CSUB-1). A macro's
 * replacement is rescanned with the tokens after it, a token once passed is never looked at again, and a name inside
 * its own expansion is painted and stays itself even where a `(` follows it later. The twin had rewritten each line
 * until it stopped changing, with no hide sets: `f(1)(2)` below came out `(1) + (2) + f` there, `g(g)(3)` `3`, `h LP
 * 1)` invoked `h`, and `EPERM` was refused. Single-valued: clang -E -P and gcc -E -P agree on every line. */
#define f(x) (x) + f
#define g(x) x
#define h(x) [x]
#define LP (
#define EPERM EPERM
#define self self + 1
#define ping pong
#define pong ping
#define twice(v) (twice + (v))
#define call(m, a) m(a)
#define id(x) x
f(1)(2)
g(g)(3)
h LP 1)
EPERM self ping pong
twice(4) call(twice, 5) call(id, id)(6)
id(id(id(7))) id(f)(8)
