/* Multi-dimensional local arrays (#localmd): `T m[A][B]` (a small grid / matrix scratch buffer) up to 3
 * dims -- lowered to a flat resource of A*B elements carrying the per-dim shape, so `m[i][j]` flattens
 * row-major to `m[i*B + j]` (declared `m[A*B]`, the same memory layout as `m[A][B]`). Both rails agree
 * + Clang-equivalent. A >3-D local defers to the LLVM fallback (the dim table holds 3). */
unsigned grid_diag(unsigned i, unsigned a) {
  unsigned m[3][4];
  for (unsigned r = 0u; r < 3u; r++)
    for (unsigned c = 0u; c < 4u; c++)
      m[r][c] = a + r * 4u + c;                 /* 2-D fill */
  return m[i % 3u][i & 3u] + m[(i + 1u) % 3u][0];   /* 2-D reads */
}
unsigned cube_sum(unsigned i, unsigned a) {
  unsigned t[2][2][2];
  unsigned s = 0u;
  for (unsigned x = 0u; x < 2u; x++)
    for (unsigned y = 0u; y < 2u; y++)
      for (unsigned z = 0u; z < 2u; z++)
        t[x][y][z] = a + x * 4u + y * 2u + z;   /* 3-D fill */
  for (unsigned x = 0u; x < 2u; x++) s += t[x][i & 1u][(i >> 1) & 1u];   /* 3-D reads */
  return s;
}
