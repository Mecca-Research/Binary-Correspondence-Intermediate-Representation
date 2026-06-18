/* compare-and-swap: the BCIR CMPXCHG opcode (a 3-read claim -- ptr, expected, desired).
 * The val form returns the pre-swap cell value; the bool form returns whether the swap won. */
uint32_t cas_update(uint32_t *cell, uint32_t expected, uint32_t desired)
{
    uint32_t old = __sync_val_compare_and_swap(cell, expected, desired);
    uint32_t won = __sync_bool_compare_and_swap(cell, desired, old);
    return old + won;
}
