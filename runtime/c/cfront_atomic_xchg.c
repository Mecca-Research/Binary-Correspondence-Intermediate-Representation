/* C11 atomic_exchange: an atomic swap -- store a new value into an _Atomic object and return its
 * prior value, in one seq-cst step. Completes the value-mutating C11 atomics (fetch_add/sub/xor +
 * load + store + exchange); emitted as the C11 function, which accepts an _Atomic*. */
uint32_t swap_slot(_Atomic uint32_t *slot, uint32_t v)
{
    uint32_t prev = atomic_exchange(slot, v);
    return prev + atomic_load(slot);
}
