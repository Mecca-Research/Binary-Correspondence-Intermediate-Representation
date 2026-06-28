/* L7 adversarial preprocessor corpus -- the no-recursion ("blue paint") rule (C 6.10.4.4).
 * A macro that (directly or indirectly) refers to itself must NOT be re-expanded once it has been
 * painted: the self-reference is left as a plain identifier. Single-valued (no impl-defined macro),
 * so clang -E -P and gcc -E -P agree exactly -- a fair single-truth differential for cpp.preprocess. */
#define A B
#define B A
#define f(x) f(x)
#define p q
#define q r
#define r p
A B
f(3)
p q r
