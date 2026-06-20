/* Local array declarators + array aggregate initializers (#localarr): `T a[N];` and `T a[N] = {...}`
 * (positional + [i]= designators) as function locals -- the scratch-buffer / DSP-accumulator idiom.
 * Completes array aggregate init on the C twin (the oracle already had it). The array local lowers to
 * a scalar-element resource of count N (emitted `T a[N]`), element access via the index load/store; an
 * aggregate initializer zero-bases (`= {0}`) then c.stores each element (gaps zero-fill, §6.7.10). */
unsigned scratch(unsigned x) { unsigned a[4]; a[x & 3u] = x; a[(x + 1u) & 3u] = x * 2u; return a[x & 3u] + a[(x + 1u) & 3u]; }
unsigned init_pos(unsigned x) { unsigned a[4] = {x, x + 1u, x + 2u}; return a[0] + a[1] + a[2] + a[3]; }
unsigned init_des(unsigned x) { unsigned a[5] = {[4] = x, [0] = x + 9u}; return a[0] * 10u + a[4] + a[2]; }
