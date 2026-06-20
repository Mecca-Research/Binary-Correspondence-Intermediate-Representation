/* Empty statements (#emptystmt): a bare `;` is a no-op -- the body of `for(...);` (all the work is in
 * the loop header) and `if(cond);`, plus stray `;;` between statements. Both rails consume it and emit
 * no claim, so behaviour is unchanged + Clang-equivalent. Previously both rejected it ("unexpected ;"). */
unsigned count_below(unsigned n) {
  unsigned i;
  for (i = 0u; i < n % 100u; i++)   /* the body is empty -- the step does the counting */
    ;
  return i;
}
unsigned masked(unsigned a) {
  unsigned r = a & 0xFFu;
  if (r > 200u)                     /* an empty-bodied guard -- a no-op either way */
    ;
  ;;                                /* stray empty statements between real ones */
  return r + (a >> 8) % 7u;
}
