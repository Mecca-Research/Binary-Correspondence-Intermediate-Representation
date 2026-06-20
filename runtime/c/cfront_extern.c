/* `extern` storage class (#extern): a declaration of a symbol defined in another translation unit --
 * the shared-config / shared-counter linkage idiom. `extern` is recognized as a storage-class
 * specifier (consumed, like `static`); an `extern T g;` global is referenced by name (no storage
 * emitted here -- it lives in the defining TU). Both rails agree structurally and the emit is
 * Clang-behaviour-equivalent when linked against the definition (supplied by the harness). */
extern unsigned cfg_base;
unsigned scaled(unsigned x) { return cfg_base * x + 1u; }
unsigned offs(unsigned x) { return (x + cfg_base) & 255u; }
void bump(unsigned d) { cfg_base += d; }
