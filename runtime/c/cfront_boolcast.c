/* Cast to `_Bool` / `bool` (#boolcast): `(_Bool)x` is `x != 0` evaluated on the FULL value (§6.3.1.2),
 * so `(_Bool)256` is 1 and `(_Bool)0.5` is 1. Both rails rendered the cast as `(uint8_t)x`, which
 * truncates to 8 bits BEFORE the bool test -- `(_Bool)256` came out 0 (256 & 0xFF), `(_Bool)0.5f` came
 * out 0 (the fraction truncated), and the twin did not normalize at all (`(_Bool)2` -> 2). The #394 fix
 * covered *stores* into a bool object; an explicit `(_Bool)` cast is a separate path. Now both rails emit
 * a real `_Bool` cast operator into a `_Bool` temp, so every conversion normalizes on the full value. */
unsigned bcast_low(unsigned a, unsigned b) {
  return (unsigned)(_Bool)(a + b);                  /* small values: nonzero -> 1 */
}
unsigned bcast_high(unsigned a, unsigned b) {
  return (unsigned)(_Bool)((a + b) << 8);            /* a value whose low byte is 0 but is still nonzero */
}
unsigned bcast_mask(unsigned a, unsigned b) {
  return (unsigned)(bool)((a & 0xFF00u) | b);        /* high-bit-only mask must still test nonzero */
}
unsigned bcast_float(unsigned a, unsigned b) {
  float x = (float)(a % 4u) + 0.5f;                  /* a fractional float: (uint8_t) would truncate to 0 */
  (void)b;
  return (unsigned)(_Bool)x;                         /* 0.5 / 1.5 / ... are all nonzero -> 1 */
}
unsigned bcast_arith(unsigned a, unsigned b) {
  return (unsigned)((_Bool)((a + b) << 16) + 10u);   /* the bool result feeds further arithmetic */
}
