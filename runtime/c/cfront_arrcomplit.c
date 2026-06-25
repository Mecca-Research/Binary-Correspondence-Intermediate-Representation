/* Array compound literals (#arrcomplit): indexed `(T[]){...}[i]`, sized with zero-fill `(T[N]){a,b}`, a
 * signed-element literal, MULTI-dim forms `(T[A][B]){{..},{..}}[i][j]` (including an inferred outer dim
 * `(T[][N]){...}`), AND 1-D AGGREGATE-element forms `(struct P[]){{..},{..}}[i].field`. Both rails lower
 * these natively: scalar multi-dim through the FLAT+shape / nested-row-brace machinery (storage `[A*B]`,
 * `[i][j]` Horner-flattens to `i*B + j` -- the #500 stride/sizing path); a struct-element literal through
 * the offset-based per-element struct store (`idx*sizeof(elem)`, riding a `= {0}` baseline), with `[i].field`
 * striding by the element struct -- the same array-of-structs descent a regular `struct P a[]` uses. All
 * gated on Clang equivalence, with the per-element-init BOUNDS-GUARD count reconciled (the twin emits a
 * masked subscript, like a regular array init, to match the oracle). (Only a MULTI-dim aggregate-element
 * literal `(struct P[A][B]){...}` remains a both-rails fallback -- the struct-leaf multi-dim stride is unmodelled.) */

struct P { unsigned x, y; };

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

unsigned agg_inferred(unsigned i) {                                                  /* 1D struct element, INFERRED */
    return (struct P[]){ {1u,2u}, {3u,4u} }[i & 1u].x;
}

unsigned agg_explicit(unsigned i) {                                                  /* 1D struct element, EXPLICIT */
    return (struct P[2]){ {5u,6u}, {7u,8u} }[i & 1u].y;
}

unsigned agg_partial(unsigned i) {                                                   /* 1D struct element, PARTIAL (= {0}) */
    return (struct P[]){ {1u}, {3u,4u} }[i & 1u].y;
}
