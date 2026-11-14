/* L7 adversarial preprocessor corpus -- a function-like macro's invocation across lines (C11 6.10.3p10, CSUB-1).
 * Within an invocation a new-line is white space: an argument list still open at the end of a line continues on the
 * next, nested parentheses and all, and a function-like macro's name that ends a line is invoked by a `(` that opens
 * the next line that is not blank. A name the next line does not open a `(` after is itself. Single-valued (no
 * implementation-defined macro, no `__LINE__` on a continued line), so clang -E -P and gcc -E -P agree exactly -- a
 * fair single-truth differential for both rails' preprocessors. */
#define ADD(a, b) ((a) + (b))
#define TRIPLE(a, b, c) [a | b | c]
#define WRAP(x) {x}
#define ID(x) x
ADD(1,
    2)
ADD(
    3,
    4
)
TRIPLE(ADD(1,
           2), (5,
                6),
       WRAP(7))
ADD
(8, 9)
ADD

  (10, 11)
WRAP(WRAP(
  WRAP(12)))
ID(ADD)(13,
  14)
ADD + 15
WRAP
- 16
ADD(17, /* a comment that spans
           lines inside an argument */ 18)
ADD("a(", ')'
    )
