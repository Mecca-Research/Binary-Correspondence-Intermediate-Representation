/* Storing a value into a `_Bool` / `bool` object (#boolnorm): C normalizes any nonzero to 1 on the
 * conversion to bool (§6.3.1.2), so `_Bool x = 2;` makes `x == 1`. The C twin emitted a bool local as
 * a plain `uint8_t`, so the store kept the raw value (2, not 1): a silent miscompile (is_clean, but the
 * differential diverged; the oracle, which emits `_Bool`, was correct). Now the twin tracks bool-ness
 * on the resource/ctype and emits `_Bool`, so every conversion-to-bool normalizes -- on a decl init, a
 * compound assignment, an array element, a parameter, and the return value. */
unsigned bool_norm(unsigned a, unsigned b) {
  _Bool x = a + b;                              /* a+b != 0 -> 1, else 0 */
  return (unsigned)x;
}
unsigned bool_mask(unsigned a, unsigned b) {
  _Bool x = (a & 0xFFu) & (b & 0xFFu);          /* a nonzero masked bit pattern still normalizes to 1 */
  return (unsigned)x * 8u;
}
unsigned bool_mod(unsigned a, unsigned b) {
  _Bool x = (a + b) % 3u;                        /* remainder 2 normalizes to 1, not 2 */
  return (unsigned)x + 5u;
}
unsigned bool_compound(unsigned a, unsigned b) {
  _Bool x = a & 1u;                              /* x is 0 or 1 */
  x += b;                                        /* the bool re-normalizes after the compound add */
  return (unsigned)x;
}
unsigned bool_array(unsigned a, unsigned b) {
  _Bool v[2];
  v[0] = a;                                      /* each element store normalizes to 0/1 */
  v[1] = b + 2u;
  return (unsigned)v[0] * 10u + (unsigned)v[1];
}
