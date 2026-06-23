/* C11 atomic_compare_exchange_strong / _weak (#cmpxchg11) -- the bool CAS whose `expected` operand is a
 * POINTER (`&exp`), the headline use of general address-of (#addrof), which is exactly why it waited on
 * `&`. Each lowers to a 3-read CMPXCHG claim (obj, &expected, desired) on lane A (R6 admits lane A for a
 * scalar atomic; R5 demands the atomic hazard), emitted verbatim as the real <stdatomic.h> generic and
 * behaviour-equivalent under Clang. The STRONG form is deterministic, so it drives the behaviour-checked
 * entry; the WEAK form may spuriously fail, so it is exercised for STRUCTURE/parity only (never the entry). */
#include <stdatomic.h>

unsigned cas_strong(_Atomic unsigned *cell, unsigned want, unsigned set)
{
    unsigned exp = want;
    if (atomic_compare_exchange_strong(cell, &exp, set))   /* &exp: the address-of payoff */
        return set + 1u;                                   /* swap won -- *cell was `want`, now `set` */
    return exp;                                            /* lost: exp now holds the observed *cell */
}

_Bool cas_weak_try(_Atomic unsigned *cell, unsigned *expected, unsigned set)
{
    return atomic_compare_exchange_weak(cell, expected, set);   /* weak: structure/parity only */
}

unsigned cmpxchg11_entry(_Atomic unsigned *cell, unsigned want, unsigned set)
{
    unsigned exp = want;
    unsigned won = atomic_compare_exchange_strong(cell, &exp, set) ? 1u : 0u;
    unsigned exp2 = exp;                                   /* exp now reflects the observed *cell */
    unsigned won2 = atomic_compare_exchange_strong(cell, &exp2, set + 3u) ? 1u : 0u;
    return won * 100u + won2 * 10u + (exp2 & 9u);
}
