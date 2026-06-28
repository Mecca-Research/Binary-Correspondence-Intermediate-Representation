/* L7 adversarial preprocessor corpus -- argument prescan and rescanning order (C 6.10.4.1).
 * An argument NOT adjacent to `#`/`##` is completely macro-expanded BEFORE substitution, then the whole
 * replacement is rescanned. So an argument that is itself a macro call expands all the way down, an
 * unparenthesized argument is substituted by token (the textbook `SQ(N+1)` surprise), and an argument
 * used several times is expanded each time. Single-valued: clang and gcc agree token-for-token. */
#define SQ(x) ((x) * (x))
#define DOUBLE(x) ((x) + (x))
#define INC(x) ((x) + 1)
#define N 3
#define VAL INC(5)
#define APPLY(f, v) f(v)
SQ(N + 1)
DOUBLE(N)
INC(VAL)
APPLY(INC, 7)
SQ(SQ(2))
