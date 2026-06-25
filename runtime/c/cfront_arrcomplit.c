/* Array compound literals (#arrcomplit): indexed `(T[]){...}[i]`, sized with zero-fill `(T[N]){a,b}`, a
 * signed-element literal, MULTI-dim SCALAR forms `(T[A][B]){{..},{..}}[i][j]` (including an inferred outer dim
 * `(T[][N]){...}`), 1-D AGGREGATE-element forms `(struct P[]){{..},{..}}[i].field`, AND MULTI-dim AGGREGATE-element
 * forms `(struct P[A][B]){{{..},{..}},{{..},{..}}}[i][j].field` (incl. an inferred outer `(struct P[][N]){...}`).
 * Both rails lower these natively: scalar multi-dim through the FLAT+shape / nested-row-brace machinery (storage
 * `[A*B]`, `[i][j]` Horner-flattens to `i*B + j` -- the #500 stride/sizing path); a struct-element literal through
 * the offset-based per-element struct store (`idx*sizeof(elem)`, riding a `= {0}` baseline), with `[i]...[k].field`
 * striding by the element struct -- the same array-of-structs descent a regular `struct P a[]` uses, the OUTER dims
 * Horner-flattened by the per-dim sizes. All gated on Clang equivalence, with the per-element-init BOUNDS-GUARD
 * count reconciled (the twin emits a masked subscript, like a regular array init, to match the oracle). */

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

unsigned md_inferred_desg(unsigned i, unsigned j) {                                  /* 2D inferred outer via DESIGNATORS */
    return (unsigned[][2]){ [1]={5u,6u}, [0]={1u,2u} }[i & 1u][j & 1u];               /* outer = max(idx)+1 = 2, NOT entry count */
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

unsigned mdagg_fixed(unsigned i, unsigned j) {                                       /* 2D struct element, FIXED dims */
    return (struct P[2][2]){ {{1u,2u},{3u,4u}}, {{5u,6u},{7u,8u}} }[i & 1u][j & 1u].x;
}

unsigned mdagg_inferred(unsigned i, unsigned j) {                                    /* 2D struct element, INFERRED outer */
    return (struct P[][2]){ {{1u,2u},{3u,4u}}, {{5u,6u},{7u,8u}}, {{9u,10u},{11u,12u}} }[i % 3u][j & 1u].y;
}

unsigned mdagg_partial(unsigned i, unsigned j) {                                     /* 2D struct element, PARTIAL (= {0}) */
    return (struct P[2][2]){ {{1u},{3u}}, {{5u,6u}} }[i & 1u][j & 1u].y;
}

unsigned mdagg_inferred_desg(unsigned i, unsigned j) {                               /* 2D struct element, INFERRED via DESIGNATORS */
    return (struct P[][2]){ [1]={{5u,6u},{7u,8u}}, [0]={{1u,2u},{3u,4u}} }[i & 1u][j & 1u].x;   /* outer = max(idx)+1 = 2 */
}
