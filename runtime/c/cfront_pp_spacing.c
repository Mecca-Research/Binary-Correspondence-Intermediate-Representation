/* L7 adversarial preprocessor corpus -- what `#` keeps of the spacing (C11 6.10.3.2p2, CSUB-1). Each run of white
 * space between an argument's tokens becomes one space in the string, and none is added elsewhere: `S(a + b)` is
 * "a + b" and `S(a+b)` "a+b". The spacing of a token that came out of a replacement is the replacement's: the first
 * token of a macro's expansion takes the macro name's, a parameter's first token the parameter's in the replacement
 * list, the commas of `__VA_ARGS__` their own, and an expansion to nothing hands its white space on to the next token.
 * Both rails had spelled a stringized argument as a line is written, "a+b" for both. Single-valued: clang -E -P and
 * gcc -E -P agree on every line. */
#define S(x) #x
#define XS(x) S(x)
#define E
#define EF()
#define P a + b
#define Q x P
#define F(a, b) S(a b)
#define G(a, b) XS(a b)
#define V(...) #__VA_ARGS__
#define CAT(a, b) a ## b
#define PL(a, b) S(a ## b c)
#define O(...) S(x __VA_OPT__(y __VA_ARGS__) z)
#define T(x) [x]
#define R(x) T( x )
#define H(p) x p
#define I(p) I2(p)
#define I2(p) p
S(a + b) S( p  q ) S(a+b) S(  )
XS(P) XS(Q) XS(a E b) XS(a E/b) XS(E a) XS(a E) XS(a(E)b)
F(1,2) F( 1 , 2 ) G(1,2) G(P,P)
V(a, b) V(a,b) V( a ,  b ) V()
PL(x, y) PL(, y) PL(x, ) XS(CAT(a, b) c) XS(CAT(,b)c)
O() O(1)
XS(R(1)) XS(T(1)T(2)) XS(S(a)b)
XS(1 +E+ 2) XS(-E-) XS(E E a E E b) XS(a EF() b) XS(x EF()/y)
XS(H()y) XS(H( )y) XS(H(E)y) XS(I()y) XS(x I(E)y) XS((I(E)))
