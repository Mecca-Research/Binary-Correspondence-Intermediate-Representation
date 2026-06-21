/* Signed comparison against an integer literal (#signedcmp): `if (x < 0)` / `x >= 0` for a signed `x` --
 * the abs / sign-test / clamp idiom, ubiquitous in driver code. The C twin emitted EVERY integer
 * constant as a hardcoded `uint32_t`, so a signed comparison against a bare literal promoted to UNSIGNED
 * (`int32_t < uint32_t` -> unsigned), making `x < 0` always false: a silent miscompile (is_clean, but
 * the differential diverged; the oracle was correct). Fixed by emitting each constant with its own
 * recorded (width, signedness), so a bare `0` is `int32_t` and the comparison stays signed. */
unsigned iabs(unsigned a, unsigned b) {
  int x = (int)(a % 2000u) - (int)(b % 2000u);
  return (unsigned)(x < 0 ? -x : x);                       /* |x| */
}
unsigned clamp_lo(unsigned a, unsigned b) {
  int x = (int)(a % 5000u) - (int)(b % 5000u);
  return x >= 0 ? (unsigned)x : 0u;                        /* clamp a negative result to 0 */
}
unsigned signum(unsigned a, unsigned b) {
  int x = (int)(a % 3000u) - (int)(b % 3000u);
  return x < 0 ? 2u : (x > 0 ? 1u : 0u);                   /* signum: 2 neg / 1 pos / 0 zero */
}
