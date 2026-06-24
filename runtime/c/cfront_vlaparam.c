/* VLA function parameters (#vlaparam): a parameter `T a[n]` decays to a pointer (C), but the runtime extent
 * `n` -- a prior in-scope integer parameter -- is RECOVERED and bound (§5.12) so the param's `a[i]` promotes
 * from assumed_safe to masked: `a[BCIR_CHK(rid, i, n, "fn:a")]`, the count re-emitted by name. Read-only over
 * the buffer (the harness runs the original then the twin on the same store, so writes would cross-talk); the
 * harness caps a scalar-with-pointer param at < 200 and backs each pointer with a 256-elem buffer, so every
 * `a[i]` with `i < n` stays in-bounds and the two rails + Clang agree. */

unsigned psum(unsigned n, unsigned a[n]) {          /* reduce over a VLA param */
    unsigned s = 0u;
    for (unsigned i = 0u; i < n; i++) s += a[i];
    return s;
}

int pdot(int n, int a[n], int b[n]) {               /* TWO VLA params, same extent -- both masked against n */
    int s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

unsigned prev(unsigned n, unsigned a[n]) {          /* index arithmetic: read a[n-1-i] (still in [0, n)) */
    unsigned s = 0u;
    for (unsigned i = 0u; i < n; i++) s += a[n - 1u - i] * (i + 1u);
    return s;
}
