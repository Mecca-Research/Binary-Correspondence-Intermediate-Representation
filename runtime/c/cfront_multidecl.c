/* Multi-declarator local declarations (#multidecl): several comma-separated declarators sharing one
 * type-specifier -- `T a = x, b, c = z;` == `T a = x; T b; T c = z;`. The canonical two-variable loop
 * init `for(unsigned i = 0u, j = n; ...)` and grouped temporaries. Each declarator lowers to its own
 * storage + (optional) copy, identical to separate declarations; both rails agree + Clang-equivalent.
 * (A per-declarator `*` / `[]` shape -- `int *p, q;` -- and the comma *operator* in a for-step are
 * follow-ons; the twin folds `*` into the specifier, so it rejects that form rather than mis-typing.) */
unsigned blend(unsigned n) {
  unsigned a = n + 1u, b = n * 3u, c = a ^ b;   /* three declarators, each initialized */
  return a + b + c;
}
unsigned windowed(unsigned n) {
  unsigned acc = 0u, scratch;                   /* one initialized, one left uninitialized */
  for (unsigned i = 0u, j = (n & 63u) + 1u; i < j; i++) /* two-variable for-init off one specifier;
                                                 * j is windowed to [1,64] so the differential harness
                                                 * (which fuzzes n up to ~2e9) doesn't run this loop a
                                                 * billion times per input -- the lowering under test is
                                                 * trip-count-independent, so the bound costs no coverage */
    acc += i & j;
  scratch = acc & 255u;
  return scratch;
}
