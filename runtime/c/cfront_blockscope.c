/* Bare-block variable scope (#blockscope): a `{ unsigned x = ...; }` block declares `x` scoped to the
 * block, so a post-block read of `x` resolves to a same-named outer -- the block must not leak into the
 * enclosing scope. Both rails save/restore the name environment around every `{ ... }` (the general case
 * of the for-loop scope fix #loopscope; a bare block lowers as an always-true `if`). */
unsigned nested_shadow(unsigned a) {
  unsigned x = a;
  {
    unsigned x = a * 2u;                 /* a nested block shadows the outer x */
    { unsigned x = a + 7u; if (x > 100u) x = 0u; }
  }
  return x;                              /* the OUTERMOST x == a */
}
unsigned block_then_use(unsigned a, unsigned b) {
  unsigned r = a;
  { unsigned r = b * 3u; if (r > 5u) r -= 1u; }   /* an inner r, discarded at the block end */
  return r + b * 0u;                     /* the OUTER r == a */
}
