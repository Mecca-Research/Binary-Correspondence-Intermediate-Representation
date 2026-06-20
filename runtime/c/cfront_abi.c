/* Target-ABI data model (#abi, the C twin of frontends/cfront/abi.py). `sizeof` folds against the
 * SELECTED target's data model: `long` follows long_size and the pointer / pointer-tracking
 * `size_t`-class types follow pointer_size, while `int` and `long long` are fixed by C across every
 * target. Under --target the five folded constants are
 *     [ sizeof long, sizeof void*, sizeof size_t, sizeof int, sizeof long long ]
 * which read [8,8,8,4,8] on LP64 (x86_64 / aarch64 / riscv64), [4,8,8,4,8] on Windows x64 LLP64,
 * and [4,4,4,4,8] on 32-bit ILP32 -- a vector that distinguishes all three data models. The default
 * (no --target) is the host LP64 model, byte-identical to the layout the frontend used before
 * --target existed, so this is also a normal host-parity fixture. A cross-target layout is not
 * byte-compatible with the host compiler, so the matrix is conformance-checked (oracle == C twin),
 * not Clang-run. */
unsigned probe(void)
{
    return (unsigned)(sizeof(long) + sizeof(void *) + sizeof(size_t)
                      + sizeof(int) + sizeof(long long));
}
