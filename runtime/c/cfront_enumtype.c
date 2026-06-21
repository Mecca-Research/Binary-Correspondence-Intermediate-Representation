/* `enum` used as a type — declaration, cast, and parameter (#enumtype): `enum Tag` is a valid
 * (int-sized) type specifier, so `enum E e;`, `(enum E)x`, and an `enum E` parameter are all legal C the
 * twin already accepted. The Python oracle's type parser set `base = "int"` for `enum [tag]` but then
 * unconditionally recomputed `base` from the (empty) keyword run, raising "expected a type" -- so it
 * rejected *every* enum-as-type, falling back where the twin compiled (a rail disagreement: the oracle
 * was stricter than the twin and Clang). Now the oracle keeps the enum's int base. Both rails lower an
 * enum to a plain int, including negative enumerators. */
enum Sign { NEG = -100, ZERO = 0, POS = 100 };

static int sign_mag(enum Sign s) { return (int)s + 7; }     /* an `enum` parameter */

unsigned e_cast(unsigned a, unsigned b) {
  enum Sign s = (enum Sign)((int)a - (int)b);               /* cast to a named enum (can be negative) */
  return (unsigned)((int)s + 1000);
}
unsigned e_select(unsigned a, unsigned b) {
  enum Sign s = a > b ? POS : (a < b ? NEG : ZERO);          /* assign named enumerators */
  return (unsigned)((int)s + 500);
}
unsigned e_param(unsigned a, unsigned b) {
  return (unsigned)(sign_mag((enum Sign)((int)a - (int)b)) + (int)b);   /* pass an enum argument */
}
