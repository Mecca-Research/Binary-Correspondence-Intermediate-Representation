/* C11 <stdatomic.h> atomics on an _Atomic object: the lock-free counter / flag pattern. The _Atomic
 * type qualifier parses and round-trips through the verified-C (emitted `_Atomic uint32_t *`), and
 * the C11 generic functions lower to the BCIR ATOMIC opcodes -- emitted as the C11 functions
 * themselves (which accept an _Atomic*; the GCC `__atomic_*` builtins do not). atomic_fetch_add/xor
 * are seq-cst read-modify-writes returning the prior value; atomic_load is an ordered read. */
uint32_t bump(_Atomic uint32_t *counter, uint32_t delta)
{
    uint32_t old = atomic_fetch_add(counter, delta);
    atomic_fetch_xor(counter, delta);
    return old + atomic_load(counter);
}
