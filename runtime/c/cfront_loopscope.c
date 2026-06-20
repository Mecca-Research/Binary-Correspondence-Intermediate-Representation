/* For-loop variable scope (#loopscope): a `for(unsigned i = ...)` declares `i` scoped to the loop, so a
 * post-loop reference to `i` resolves to a same-named param / outer local -- the for-init is a fresh
 * scope, NOT leaked into the enclosing block. Both rails save/restore the name environment around the
 * loop; without it, `return s + i` below would read the loop counter (5 / the last index) -- a miscompile
 * the duplicate-name emit fix (#loopreuse) had turned from a build error into a silent wrong answer. */
unsigned param_shadow(unsigned i) {
  unsigned s = 0u;
  for (unsigned i = 0u; i < 5u; i++) s += i;        /* loop i (0..4) shadows the param i */
  return s + i;                                     /* the PARAM i, not the loop counter */
}
unsigned outer_shadow(unsigned n) {
  unsigned i = 100u, s = 0u;                        /* an outer i */
  for (unsigned i = 0u; i < n % 6u; i++) s += i * 2u;  /* the inner i shadows it within the loop */
  return s + i;                                     /* the OUTER i == 100 */
}
