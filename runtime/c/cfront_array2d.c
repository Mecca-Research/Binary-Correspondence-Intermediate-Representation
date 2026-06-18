/* Multi-dimensional arrays (L3): a 2D array parameter `uint32_t m[4][8]` decays to a flat element
 * pointer, and a multi-index access `m[i][j]` flattens row-major to the linear index `i*8 + j`
 * (Horner) -- reusing the 1D pointer/index/load machinery on both rails. Indices are masked so the
 * access stays inside the row/column extents. The canonical register-bank / tile-buffer pattern. */
uint32_t tile_sum(uint32_t m[4][8], uint32_t i, uint32_t j)
{
    uint32_t a = m[i & 3][j & 7];
    uint32_t b = m[(i + 1) & 3][j & 7];
    return a + b;
}
