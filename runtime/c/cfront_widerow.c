/* Array-of-row pointer declarator `(*m)[N]` (L3): a parameter `uint32_t (*m)[8]` is a pointer to a row
 * of 8 -- exactly what a 2D array `m[][8]` decays to -- so `m[i][j]` flattens row-major to the linear
 * index `i*8 + j`, reusing the multi-dimensional array machinery on both rails. This is the remaining
 * vendor-header declarator form (register banks / tile buffers passed as a row pointer). Indices are
 * masked to stay inside the backing buffer. Clang-behaviour-equivalent. */
uint32_t row_sum(uint32_t (*m)[8], uint32_t i, uint32_t j)
{
    uint32_t a = m[i & 7][j & 7];
    uint32_t b = m[(i + 1) & 7][j & 7];
    return a + b;
}
