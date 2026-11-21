/* L7 adversarial preprocessor corpus -- rescanning and pasting (C11 6.10.3.3, 6.10.3.4; CSUB-1).
 * A replacement is rescanned *along with the tokens after it*: a replacement that ends in a function-like macro's
 * name is invoked by a `(` that follows (`G(1)`), and a replacement may take its arguments from what follows
 * (`LP 1)`). An operand of `##` is the RAW argument (`CAT(X, 2)` is `X2` whatever `X` names), and an empty argument
 * is a placemarker: it pastes into nothing and never glues the tokens around it (`x a y` of an empty `a` is `x y`).
 * A parameter only stringized is never macro-replaced (`S(F2(1))`). Single-valued: clang -E -P and gcc -E -P agree. */
#define F(x) ((x) + 1)
#define G F
#define H(y) F
#define LP F(
#define CAT(a, b) a##b
#define X 1
#define Y 2
#define E(a) x a y
#define P(a, b) x a##b
#define Q(a, b) a##b y
#define K(a, b, c) p a##b##c q
#define S(x) #x
#define F2(a, b) a b
#define ff(a) a * gg
#define gg(a) ff(a)
#define GG F(1) F
G(1)
H(0)(2)
LP 3)
ff(2)(9)
GG(4)
CAT(X, 2) CAT(X, Y) CAT(, Y) CAT(X, )
E() P(, y) P(, ) Q(q, ) Q(, )
K(, , ) K(1, , ) K(, , 3) K(, 2, )
S(F2(1))
