/* `_Thread_local` / `thread_local` storage class (#threadlocal): per-thread storage (kernel per-CPU
 * vars, TLS counters). Recognized as a storage-class specifier and consumed (like static/extern); the
 * value model is the deterministic single-thread executor, so a thread-local global behaves as a global
 * here -- referenced by name, defined in the source / harness. Both rails agree + Clang-equivalent. */
_Thread_local unsigned tls_ctr;
unsigned scaled(unsigned x) { return tls_ctr * x + 1u; }
unsigned offs(unsigned x) { return (x + tls_ctr) & 255u; }
void bump(unsigned d) { tls_ctr += d; }
