/* Scalar file-scope globals: read + write (#globals). The oracle models a scalar global as a plain
 * resource -- a read references it directly (no c.load), a write is a c.copy to it. `seed` writes the
 * global, `peek` only reads it (zero claims, a bare return of the global rid), and `accumulate` does a
 * plain write + a compound read-modify-write + a read. accumulate is a non-pure function (it mutates
 * module-scope state), so the dual-rail behaviour check resets the global between the original and the
 * emitted twin (a side-effect-aware harness) rather than the generic pure-function one. Parity +
 * Clang behaviour-equivalence. */
static unsigned acc;

void seed(unsigned v)
{
    acc = v;                 /* a plain global write -> c.copy to the global rid */
}

unsigned peek(void)
{
    return acc;              /* a bare global read -> return_rid is the global (no claim) */
}

unsigned accumulate(unsigned a, unsigned b)
{
    acc = a;                 /* write */
    acc += b;                /* compound read-modify-write: c.bin.add + c.copy */
    return acc + 1u;         /* read the global in an expression */
}
