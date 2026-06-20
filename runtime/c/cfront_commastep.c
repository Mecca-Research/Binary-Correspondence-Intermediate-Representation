/* The comma operator in its dominant position -- the for-loop step (#commastep): `for(...; ...; i++,
 * j--)`. Two-pointer / reversal loops and parallel-counter updates. Each comma-separated step element
 * (inc/dec, plain or compound assignment) runs in order at the end of every iteration; both rails parse
 * the list -- the twin re-parses the recorded step tokens, looping on the commas -- + Clang-equivalent. */
unsigned span_xor(unsigned n) {            /* walk two cursors toward the middle (i up, j down) */
  unsigned acc = 0u;
  for (unsigned i = 0u, j = n | 1u; i < j; i++, j--)
    acc ^= (i * 3u) + j;
  return acc;
}
unsigned twin_acc(unsigned n) {            /* two accumulators advanced together each iteration */
  unsigned s = 0u, a = 0u, b = 1u;
  for (unsigned i = 0u; i < n % 24u; i++, a += b, b += a & 7u)
    s += a;
  return s;
}
