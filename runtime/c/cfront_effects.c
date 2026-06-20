/* Module-scope effect / commutation analysis (#effects). Two file-scope globals, ga and gb, let the
 * analysis exercise every case: read_a / read_b read disjoint globals (they COMMUTE); write_a writes
 * ga (it conflicts with read_a / via_a, but commutes with read_b which only touches gb); via_a calls
 * write_a then read_a, so its folded footprint is {reads ga, writes ga} (callee effects fold into the
 * caller transitively). The C twin's `own_footprint`/commute (bcir_cfront_effects) must agree with the
 * oracle's pipeline.effects / commute for every function and every pair. No string literals (their
 * global names aren't shared across the two rails). */
static unsigned ga;
static unsigned gb;

unsigned read_a(void)        { return ga; }              /* reads ga */
unsigned read_b(void)        { return gb; }              /* reads gb -> commutes with read_a */
void     write_a(unsigned v) { ga = v; }                 /* writes ga */

unsigned via_a(unsigned v)                               /* folds write_a (W ga) + read_a (R ga) */
{
    write_a(v);
    return read_a();
}
