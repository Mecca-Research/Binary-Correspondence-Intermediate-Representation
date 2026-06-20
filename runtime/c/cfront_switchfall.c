/* Cross-clause `switch` fallthrough (#switchfall): a real C switch -- a `case` without `break` falls
 * into the next clause, `default` participates, and stacked labels share a body. Both rails now emit a
 * genuine `switch(d){ case V: ... }` (case labels fold to constants; the discriminant is lowered once;
 * `break` is preserved), so fallthrough behaves exactly as C specifies (the old if/else-if desugar
 * ran only one clause). Returns values, so the generic equivalence harness applies. */
unsigned classify(unsigned op) {
  unsigned r = 0u;
  switch (op) {
    case 0u: r += 1u;          /* falls through */
    case 1u: r += 10u; break;
    case 2u:
    case 3u: r += 100u;        /* stacked labels -> shared body, falls through */
    default: r += 1000u;
  }
  return r;
}
unsigned ladder(unsigned n) {
  unsigned acc = 0u;
  switch (n) {
    case 4u: acc += 4u;
    case 3u: acc += 3u;
    case 2u: acc += 2u;
    case 1u: acc += 1u;        /* a classic descending fallthrough ladder */
  }
  return acc;
}
