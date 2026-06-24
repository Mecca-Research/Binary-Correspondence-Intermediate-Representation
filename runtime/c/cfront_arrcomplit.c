/* 1-D SCALAR-element array compound literals (#arrcomplit): indexed `(T[]){...}[i]`, sized with zero-fill
 * `(T[N]){a,b}`, and a signed-element literal. Both rails lower these natively. This pins them -- including
 * the per-element-init BOUNDS-GUARD count, which the twin now emits as a masked subscript (like a regular
 * array init) to match the oracle. (Multi-dim `(T[a][b]){...}` and aggregate-element `(struct P[]){...}` stay
 * a both-rails fallback -- a deferred reconciliation follow-on.) */

unsigned idx1d(unsigned i) { return (unsigned[]){ 10u, 20u, 30u, 40u }[i & 3u]; }   /* 1D indexed */

unsigned sized1d(unsigned i) {                                                       /* 1D sized + zero-fill */
    return (unsigned[4]){ 10u, 20u }[i & 3u];
}

int signed1d(unsigned i) {                                                           /* a signed-element 1D literal */
    return (int[]){ -1, -2, -3 }[i % 3u];
}
