/* goto + labels (L6): the driver error-cleanup pattern -- forward `goto done;` jumps to a label at
 * function-body scope. Both rails carry the jump + target as emit-only markers (like break/continue,
 * which already lower to a goto), so they emit `goto done;` / `done:;` verbatim and stay Clang-
 * behaviour-equivalent. The mutable accumulator `r` is a real C local, so the skipped updates match. */
uint32_t classify(uint32_t x, uint32_t y)
{
    uint32_t r = 1;
    if (x == 0) goto done;
    r = r + x;
    if (y == 0) goto done;
    r = r * y;
done:
    return r;
}
