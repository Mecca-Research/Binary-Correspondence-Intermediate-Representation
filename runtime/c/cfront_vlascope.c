/* A block that declares a stack VLA is emitted as a scope of its own (CF-VLASCOPE). The emit inlines every other
 * block, and a VLA inlined into its parent put the labels after the block inside the array's scope -- a later
 * `case`, a loop's continue label, a `goto` target -- which C forbids a jump to enter (C11 6.8.4.2p2, 6.8.6.1p1):
 * the emit of each function below did not compile. Sizes come from the input and every access stays in bounds,
 * so the two rails and Clang agree for every input; `vlascope` (the entry) runs them all. */

unsigned vs_case(unsigned n, unsigned x) {   /* a VLA in a braced case arm, then more labels */
    unsigned s = 7u;
    switch (x & 3u) {
    case 1: {
        unsigned a[n];
        a[0] = x;
        a[n - 1u] = x + 1u;
        s = a[0] + a[n - 1u];
        break;
    }
    case 2:
        s = x * 2u;
    default:
        s = s + 1u;
    }
    return s;
}

unsigned vs_continue(unsigned n, unsigned c) {   /* a `continue` before a VLA declared in the loop body */
    unsigned s = 0u;
    while (c > 0u) {
        c = c - 1u;
        if ((c & 1u) != 0u) continue;
        unsigned a[n];
        a[0] = c;
        s = s + a[0];
    }
    return s;
}

unsigned vs_for(unsigned n, unsigned c) {   /* the same in a `for`, whose continue point runs the step */
    unsigned s = 0u;
    for (unsigned i = 0u; i < (c & 7u); i++) {
        if (i == 2u) continue;
        unsigned a[n];
        a[n - 1u] = i * 5u;
        s = s + a[n - 1u];
    }
    return s;
}

unsigned vs_goto(unsigned n, unsigned c) {   /* a `goto` past a braced block that declares a VLA */
    if (c > 9u) goto out;
    {
        unsigned a[n];
        a[0] = c + 4u;
        c = a[0];
    }
out:
    return c;
}

unsigned vs_nested(unsigned n, unsigned m) {   /* two VLAs in one block, one of two dimensions, an inner block too */
    unsigned s = 0u;
    for (unsigned i = 0u; i < 4u; i++) {
        if (i == 3u) break;
        unsigned a[n];
        a[0] = i;
        unsigned b[n][m];
        b[0][0] = a[0] + 1u;
        {
            unsigned c[m];
            c[m - 1u] = b[0][0];
            s = s + c[m - 1u];
        }
    }
    return s;
}

unsigned vs_inner(unsigned n, unsigned x) {   /* only an inner block declares a VLA: the loop body stays inline */
    unsigned s = 0u;
    for (unsigned i = 0u; i < 3u; i++) {
        s = s + i;
        {
            unsigned a[n];
            a[0] = x + i;
            s = s + a[0];
        }
    }
    return s;
}

unsigned vs_if(unsigned n, unsigned x) {   /* a VLA in an `if` branch, the branch braced already */
    if (x != 0u) {
        unsigned a[n];
        a[0] = x;
        x = a[0] * 3u;
    }
    return x;
}

unsigned vlascope(unsigned x, unsigned y) {
    unsigned n = (x & 7u) + 1u, m = (y & 3u) + 1u;
    return vs_case(n, y) + vs_continue(n, y & 15u) + vs_for(n, y) + vs_goto(n, y & 15u) + vs_nested(n, m)
           + vs_inner(n, x) + vs_if(n, x);
}
