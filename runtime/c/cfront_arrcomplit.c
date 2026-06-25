/* SCALAR-element array compound literals (#arrcomplit): indexed `(T[]){...}[i]`, sized with zero-fill
 * `(T[N]){a,b}`, a signed-element literal, and MULTI-dim forms `(T[A][B]){{..},{..}}[i][j]` (including an
 * inferred outer dim `(T[][N]){...}`). Both rails lower these natively through the FLAT+shape / nested-row-
 * brace machinery, so the multi-dim storage is `[A*B]` (the right inner stride) and `[i][j]` Horner-flattens
 * to `i*B + j` -- the #500 stride/sizing path, gated on Clang equivalence. This pins them -- including the
 * per-element-init BOUNDS-GUARD count, which the twin emits as a masked subscript (like a regular array init)
 * to match the oracle. (Aggregate-element `(struct P[]){...}` stays a both-rails fallback -- the final follow-on.) */

unsigned idx1d(unsigned i) { return (unsigned[]){ 10u, 20u, 30u, 40u }[i & 3u]; }   /* 1D indexed */

unsigned sized1d(unsigned i) {                                                       /* 1D sized + zero-fill */
    return (unsigned[4]){ 10u, 20u }[i & 3u];
}

int signed1d(unsigned i) {                                                           /* a signed-element 1D literal */
    return (int[]){ -1, -2, -3 }[i % 3u];
}

unsigned md_fixed(unsigned i, unsigned j) {                                          /* 2D fixed dims */
    return (unsigned[2][2]){ {1u,2u}, {3u,4u} }[i & 1u][j & 1u];
}

unsigned md_inferred(unsigned i, unsigned j) {                                       /* 2D inferred outer dim */
    return (unsigned[][2]){ {1u,2u}, {3u,4u}, {5u,6u} }[i % 3u][j & 1u];
}

int md_signed(unsigned i, unsigned j) {                                              /* a signed-element 2D literal */
    return (int[2][2]){ {-1,-2}, {-3,-4} }[i & 1u][j & 1u];
}
