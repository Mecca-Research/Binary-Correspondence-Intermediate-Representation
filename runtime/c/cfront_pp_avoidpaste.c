/* L7 adversarial preprocessor corpus -- tokens that must NOT paste (C 6.4p4, maximal munch; CF-PASTE).
 * A preprocessor that re-spells a line from its tokens must keep a space between two tokens whose
 * spellings, run together, lex as others: `a + ++g` written `a+++g` is `(a++) + g`, `-NEG(a)` with
 * `NEG(x) -x` written `--a` is a decrement, and `y / *p` written without its space opens a comment. Both
 * cfront preprocessors re-spelled every line and kept a space only between two words, so each of these
 * came out as other tokens. The source's own spellings, object and function macro expansions, three
 * tokens that make one (`. . .`), and `##`'s deliberate pastes (which must still glue) are all here; so
 * are spellings that maximal munch already splits (`i-- > 0`), which must keep their text. Single-valued:
 * clang and gcc agree. */
#define NEG(x) -x
#define INC ++
#define PLUS +
#define MINUS -
#define LT <
#define DOT .
#define DIV /
#define STAR *
#define ONE 1
#define E 1e
#define ID(x) x
#define CAT(a, b) a ## b
a + ++g
a - --g
a + +b
a - -b
y / *p
y / /p
-NEG(a)
c + INC b
PLUS PLUS PLUS
MINUS MINUS> x
LT LT= y
LT LT LT
DIV STAR c STAR DIV
x DOT DOT DOT y
ONE DOT 5
ONE ... ONE
E+1
0x1e + 1
ID(y) / ID(*p) + ID(-)-y
i-- > 0
x++ == y
s->x-->0
CAT(+, +) CAT(-, >) CAT(<, <=)
x * .5f + 1.5e+3f
