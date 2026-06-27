/* C23 `_BitInt(N)` (#bitint): bit-precise integers as locals, parameters, and return types, with
 * same-type arithmetic that wraps at the EXACT width N (no integer promotion) -- the headline being a
 * NON-standard lane width (`_BitInt(12)` / `_BitInt(20)`) that has no fixed-width type. The frontend
 * carries N exactly and EMITS `_BitInt(N)` / `unsigned _BitInt(N)` faithfully, so Clang applies the
 * N-bit semantics in BOTH the original source and the re-emitted bcir_* -- which is what keeps them
 * behaviour-equivalent. Only SAME-type `_BitInt(N)` arithmetic (and `_BitInt` with integer constants)
 * is in the supported subset; mixing widths / standard ints routes to fallback (see test_c_cfront.py).
 *
 * Arithmetic is kept on UNSIGNED `_BitInt` lanes (wrap-around is well defined, so the equivalence harness
 * may seed the full value range without signed-overflow UB); a signed `_BitInt` is exercised by a
 * width-narrowing read-back, which is conversion (defined), not signed overflow.
 *
 * The entry function (the LAST one, which the equivalence harness runs) RETURNS a computed `_BitInt(12)`
 * value, so Clang runs both binaries on seeded inputs and compares the N-bit-wrapped results. */

/* an unsigned 12-bit lane: a + b wraps mod 2^12 (computed at 12 bits, NOT promoted to int) */
unsigned _BitInt(12) bi12_add(unsigned _BitInt(12) a, unsigned _BitInt(12) b) {
  unsigned _BitInt(12) c = a + b;            /* wraps at 12 bits */
  return c;
}

/* an unsigned 20-bit lane: same-type multiply + a constant, staying `_BitInt(20)` (wrap-around defined) */
unsigned _BitInt(20) bi20_macc(unsigned _BitInt(20) a, unsigned _BitInt(20) b) {
  unsigned _BitInt(20) p = a * b;            /* the product stays 20-bit (wraps), no promotion */
  unsigned _BitInt(20) q = p - (unsigned _BitInt(20))7;   /* a `_BitInt` with an integer constant stays `_BitInt(20)` */
  return q;
}

/* an unsigned 64-bit `_BitInt` (a width that DOES coincide with a standard type, but stays distinct):
 * bitwise + shift on same-type operands. */
unsigned _BitInt(64) bi64_bits(unsigned _BitInt(64) a, unsigned _BitInt(64) b) {
  unsigned _BitInt(64) m = (a & b) | (a ^ b);
  unsigned _BitInt(64) s = m << (unsigned _BitInt(64))3;
  return s + a;
}

/* a SIGNED `_BitInt(20)`: the conversion of an unsigned value into a narrower signed lane is defined
 * (implementation-defined result, but NOT UB), and the read-back through `- self` cancels to a known
 * value -- so this exercises the signed `_BitInt` spelling + non-promotion without signed overflow. */
unsigned _BitInt(12) bi_signed20(unsigned _BitInt(12) a) {
  _BitInt(20) s = (_BitInt(20))a;            /* widen the 12-bit value into a signed 20-bit lane */
  _BitInt(20) t = s + (_BitInt(20))1 - (_BitInt(20))1;    /* same-type signed add/sub (no overflow): == s */
  return (unsigned _BitInt(12))t;            /* narrow back to the 12-bit result */
}

/* the ENTRY: compose the lanes through a 12-bit result the harness compares. Each intermediate value
 * stays its own `_BitInt(N)` width; the final return is `_BitInt(12)`, so every step is exercised by
 * Clang's N-bit arithmetic on both rails. */
unsigned _BitInt(12) bitint_entry(unsigned _BitInt(12) a, unsigned _BitInt(12) b) {
  unsigned _BitInt(12) s = bi12_add(a, b);   /* 12-bit add (wraps) */
  unsigned _BitInt(12) t = s * a + b;        /* same-type 12-bit mul/add (wraps) */
  unsigned _BitInt(12) u = t - (unsigned _BitInt(12))5 + bi_signed20(a);
  return u;
}
