/* Nested-brace initializers for multi-dimensional local arrays (#localmdinit): `T a[A][B] = {{..},{..}}`
 * up to 3 dims -- the flat resource of A*B elements (carrying the per-dim shape, same row-major layout as
 * `a[A][B]`) is initialized by descending each outer brace by ROW: row r lands at offset r*stride, the
 * innermost dim stores per element. Rides the `= {0}` zero baseline (§6.7.10), so a partial init zero-fills.
 * Both rails agree + Clang-equivalent. Indexed `a[i][j]` (the existing Horner read). */
unsigned grid2(unsigned i, unsigned j) {
  unsigned a[2][2] = {{1u, 2u}, {3u, 4u}};      /* 2-D nested-brace init */
  return a[i & 1u][j & 1u];
}
unsigned cube3(unsigned i, unsigned j, unsigned k) {
  unsigned b[2][2][2] = {{{1u, 2u}, {3u, 4u}}, {{5u, 6u}, {7u, 8u}}};   /* 3-D nested-brace init */
  return b[i & 1u][j & 1u][k & 1u];
}
unsigned partial(unsigned i, unsigned j) {
  unsigned c[2][3] = {{1u}, {4u, 5u}};          /* partial init -> the `= {0}` zero baseline fills the gaps */
  return c[i & 1u][j % 3u];
}
