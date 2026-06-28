/* C23 `_BitInt(N)` MIXED-WIDTH arithmetic (#bitintmixed): a `_BitInt(N)` mixed in one expression with a
 * NARROWER standard integer VARIABLE, a NARROWER `_BitInt`, or a narrower integer constant -- the cases
 * whose C23 6.3.1.8 usual-arithmetic-conversions result is itself a `_BitInt(N)` (the bit-precise operand
 * WINS the rank, its width strictly exceeding the other operand's post-promotion width). The frontend
 * MODELS that result type exactly (verified == Clang by the `_Generic` differential in test_c_cfront.py)
 * and EMITS both operands with their faithful spellings, so Clang applies the same conversions in BOTH the
 * original source and the re-emitted bcir_* -- keeping them behaviour-equivalent.
 *
 * The headline is that the result keeps the EXACT non-standard lane width: `unsigned _BitInt(40)` + a
 * `uint32_t` stays `unsigned _BitInt(40)` (NOT widened to a power-of-two type), so the 40-bit wrap is
 * preserved. Arithmetic is kept on UNSIGNED lanes (wrap-around is defined, so the equivalence harness may
 * seed the full value range without signed-overflow UB). The entry RETURNS a computed `_BitInt(40)`.
 *
 * STAYS FALLBACK (NOT here): a `_BitInt(N)` mixed with an EQUAL-OR-WIDER standard int / constant, where the
 * C23 result is a STANDARD type (e.g. `_BitInt(8) + int` -> `int`) -- see test_cfront_fallback.py. */

/* a 40-bit lane mixed with a 32-bit standard variable: `_BitInt(40)` wins (40 > 32), result `_BitInt(40)` */
unsigned _BitInt(40) bi40_mix_int(unsigned _BitInt(40) a, unsigned b) {
  unsigned _BitInt(40) c = a + b;            /* ubi40 + uint -> ubi40 (the wider bit-int wins) */
  return c;                                  /* the 40-bit wrap is preserved */
}

/* a 64-bit lane mixed with a 32-bit standard variable, and a bare narrower constant */
unsigned _BitInt(64) bi64_mix(unsigned _BitInt(64) a, unsigned b) {
  unsigned _BitInt(64) p = a * b;            /* ubi64 * uint -> ubi64 */
  unsigned _BitInt(64) q = p + 7u;           /* ubi64 + (unsigned)7 -> ubi64 (the constant is narrower) */
  return q;
}

/* a WIDER `_BitInt` mixed with a NARROWER `_BitInt`: the wider (here 40-bit) wins */
unsigned _BitInt(40) bi_widen(unsigned _BitInt(40) a, unsigned _BitInt(20) b) {
  unsigned _BitInt(40) m = a | (unsigned _BitInt(40))b;   /* widen the 20-bit lane, OR into the 40-bit one */
  unsigned _BitInt(40) n = a + b;            /* ubi40 + ubi20 -> ubi40 directly (the mix is modeled) */
  return m ^ n;
}

/* the ENTRY: compose the lanes through a 40-bit result the harness compares. Every intermediate keeps its
 * own `_BitInt(N)` width; the final return is `_BitInt(40)`, exercised by Clang's N-bit arithmetic on both
 * rails. The standard-int parameter `k` mixes in below its bit-int's rank, so the result stays bit-int. */
unsigned _BitInt(40) bitint_mixed_entry(unsigned _BitInt(40) a, unsigned _BitInt(20) b, unsigned k) {
  unsigned _BitInt(40) s = bi40_mix_int(a, k);            /* ubi40 + uint -> ubi40 */
  unsigned _BitInt(40) t = bi_widen(a, b);                /* a 40/20 bit-int mix -> ubi40 */
  unsigned _BitInt(40) u = s + t * (unsigned _BitInt(40))3 + k;   /* same-type mul, then ubi40 + uint -> ubi40 */
  return u;
}
