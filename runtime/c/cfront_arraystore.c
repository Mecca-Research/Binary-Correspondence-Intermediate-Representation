/* Array element stores (#astore): a[i] = v and a[i] OP= v -- the driver buffer-fill / scatter / DSP
 * idiom (writing through a pointer/array parameter). Lowers to a 3-read c.store (base, index, value),
 * the twin of the oracle's indexed _write; the emit renders a[i] = v. Both rails agree structurally
 * and the emitted C is Clang-behaviour-equivalent (the harness writes distinct buffers per rail). */
void fill(unsigned *a, unsigned n, unsigned base) { a[n] = base; a[n + 1u] = base + 1u; }
void scatter(unsigned *a, unsigned i, unsigned j, unsigned v) { a[i] = v; a[j] = v + 7u; }
void accum(unsigned *a, unsigned i, unsigned d) { a[i] += d; a[i] |= 1u; a[i] *= 2u; }
