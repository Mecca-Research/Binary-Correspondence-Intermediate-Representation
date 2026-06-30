/* atomics + fences: the BCIR ATOMIC and BARRIER opcodes (the atomic-hazard R5 contract).
 * SEG6.1/SEG7 -- the order-parameterized fence forms (`__atomic_thread_fence(order)` and the C11
 * `atomic_thread_fence(order)`) parse their `memory_order` ARGUMENT and route to the fence KIND it
 * implies (acquire / release / full), with the unshadowed `memory_order_*` / `__ATOMIC_*` name read as
 * its int const -- byte-identical on BOTH rails, so the dual-rail digest gate verifies the lowering
 * symmetry. The named constants need no <stdatomic.h> include: the frontend/oracle recognize them
 * natively, and the behaviour harness adds the header for the real-compiler differential. */
uint32_t atomic_inc(uint32_t *counter, uint32_t delta)
{
    uint32_t old = __atomic_fetch_add(counter, delta, 5);
    __atomic_thread_fence(5);                       /* full fence -- integer order arg (backward-compat) */
    atomic_thread_fence(memory_order_acquire);      /* C11 ordered: acquire -> c.fence.acquire */
    atomic_thread_fence(memory_order_release);      /* C11 ordered: release -> c.fence.release */
    atomic_thread_fence(memory_order_seq_cst);      /* C11 ordered: seq_cst -> full c.fence */
    __atomic_thread_fence(__ATOMIC_ACQUIRE);        /* GCC ordered: acquire -> c.fence.acquire */
    __atomic_thread_fence(__ATOMIC_RELEASE);        /* GCC ordered: release -> c.fence.release */
    atomic_thread_fence((memory_order_acquire));    /* PARENTHESIZED order (e.g. a macro-expanded `#define ACQ
                                                       (memory_order_acquire)`): both rails strip redundant
                                                       parens -> c.fence.acquire (pins the paren-stripping parity) */
    __atomic_thread_fence(+2);                      /* leading unary `+` (e.g. a macro-expanded `#define ORD +2`):
                                                       both rails drop the transparent `+` -> c.fence.acquire
                                                       (pins the unary-plus parity) */
    uint32_t prev = __atomic_fetch_xor(counter, old, 5);
    return old + prev;
}
