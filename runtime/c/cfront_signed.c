/* The `signed` type specifier (#signedty): `signed char` / `signed int` / `signed long` as a declarator
 * and a cast. The C twin recognized `unsigned` but not `signed` in its declaration- and cast-type-start
 * detection, so `signed char sc = (signed char)a` was rejected (-> fallback) while the oracle accepted
 * it: a rail disagreement. Now `signed` is a recognized type-start on both rails (a modifier -- the base
 * sets the width; `signed` alone, i.e. signed int with no base, stays a fallback on both). */
unsigned signed_roundtrip(unsigned a) {
  signed char sc = (signed char)a;             /* narrow to signed char ... */
  signed int si = (signed int)(a * 7u);
  signed long sl = (signed long)(a + 3u);
  return (unsigned)(int)sc + (unsigned)si + (unsigned)sl;   /* ... and widen back (sign-extend) */
}
unsigned signed_scale(unsigned a, unsigned b) {
  signed int x = (signed int)a;
  signed int y = (signed int)(b + 1u);
  return (unsigned)(x * y - x + y);            /* signed wrapping arithmetic (no sign-dependent branch) */
}
