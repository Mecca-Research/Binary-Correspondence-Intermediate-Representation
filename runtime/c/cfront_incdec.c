/* Increment / decrement: i++ / ++i / i-- / --i, in statements and for-loop clauses (the value is
 * discarded). Each desugars to `i = i ± 1` (a const + a binary op + a copy) identically on both
 * rails -- the ubiquitous loop-counter / accumulator idiom. */
uint32_t count(uint32_t n)
{
    uint32_t s = 0;
    uint32_t d = n;
    for (uint32_t i = 0; i < n; i++) {
        s++;
        --d;
    }
    return s + d;
}
