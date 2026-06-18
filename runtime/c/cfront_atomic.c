/* atomics + a fence: the BCIR ATOMIC and BARRIER opcodes (the atomic-hazard R5 contract). */
uint32_t atomic_inc(uint32_t *counter, uint32_t delta)
{
    uint32_t old = __atomic_fetch_add(counter, delta, 5);
    __atomic_thread_fence(5);
    uint32_t prev = __atomic_fetch_xor(counter, old, 5);
    return old + prev;
}
