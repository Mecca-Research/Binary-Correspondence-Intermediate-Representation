/* Reused local names across scopes (#loopreuse): two separate `for` loops each declaring `unsigned i`
 * are distinct locals (disjoint source scopes) that flatten to distinct rids carrying the same name.
 * The emit must give each a UNIQUE C identifier (`i`, `i_2`, ...) -- declaring both at function scope
 * would be a C redefinition (the unit was is_clean yet its emit did not compile). Behaviour matches
 * Clang: a fresh, re-initialized variable per loop. A naive dedup would corrupt a shadowed value. */
unsigned two_pass(unsigned n) {
  unsigned acc = 0u;
  for (unsigned i = 0u; i < n % 13u; i++) acc += i;        /* first i */
  for (unsigned i = 0u; i < n % 7u; i++) acc += i * 3u;    /* second i -- same name, separate scope */
  return acc;
}
unsigned triple_pass(unsigned n) {
  unsigned r = 1u;
  for (unsigned k = 0u; k < n % 5u; k++) r += k;
  for (unsigned k = 1u; k < n % 4u; k++) r *= (k + 1u);    /* k reused */
  for (unsigned k = 0u; k < n % 3u; k++) r -= k;           /* k reused again */
  return r;
}
