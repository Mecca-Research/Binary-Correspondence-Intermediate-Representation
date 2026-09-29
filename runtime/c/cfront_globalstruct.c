/* CF-GSTRUCT / CF-GINIT: a file-scope struct's members, read and written every way a local's are. The twin
 * bound a struct global with struct index -1, so every member access read `c->s[-1]` and crashed the
 * frontend (a store through `gs.n = v` too, a nested `gt.in.a[1]`, a pointer global's `gp->n`); its device
 * domain was computed without the struct, so a global holding volatile members was ordinary memory. The
 * oracle typed an INITIALIZED scalar or struct global as a one-element array of it -- a relic of the
 * read-only lookup-table model -- so `gw + 1` was computed in 32 bits (`0x100000000 + 1` returned 1) and
 * `gi.n` was refused. Each function reads or writes the globals below; the test seeds them, runs the
 * original and each emit from the same state, and compares the result and every global's bytes. */
#include <stdint.h>

struct gs_s { uint8_t c; uint64_t n; uint32_t a[4]; };
struct gs_t { uint16_t k; struct gs_s in; uint8_t z[3]; };
struct gs_v { uint32_t plain; volatile uint32_t regs[4]; };

struct gs_s gs_g;
struct gs_t gs_t2;
struct gs_s gs_arr[3];
static struct gs_s gs_static;
struct gs_s *gs_p;
struct gs_v gs_dev;
uint64_t gs_wide = 0x100000000u;
int8_t gs_small = -3;
struct gs_s gs_init = {1, 0x100000005u};

uint64_t gs_read_u64(void) { return gs_g.n; }
uint32_t gs_read_u8(void) { return gs_g.c; }
void gs_write(uint64_t v) { gs_g.n = v; gs_g.c = (uint8_t)v; }
uint32_t gs_member_array(uint32_t i) { gs_g.a[i & 3u] = i * 7u; return gs_g.a[(i + 1u) & 3u] + gs_g.a[3]; }
uint32_t gs_nested(uint32_t v) { gs_t2.in.c = (uint8_t)v; gs_t2.in.a[2] = v + 1u; return gs_t2.in.a[1] + gs_t2.z[v % 3u]; }
uint64_t gs_compound(uint64_t v) { gs_g.n += v; gs_g.n++; return gs_g.n = gs_g.n * 3u; }
uint64_t *gs_address(void) { return &gs_g.n; }
struct gs_s gs_whole(void) { return gs_g; }
void gs_whole_write(struct gs_s v) { gs_g = v; }
uint64_t gs_through_pointer(void) { struct gs_s *p = &gs_g; return p->n + gs_t2.in.n; }
void gs_array_of_structs(uint32_t i, uint64_t v) { gs_arr[i % 3u].n = v; gs_arr[(i + 1u) % 3u].c = (uint8_t)i; }
uint64_t gs_static_member(void) { gs_static.n = gs_static.n + 1u; return gs_static.n; }
uint64_t gs_pointer_global(void) { return gs_p ? gs_p->n + gs_p->a[1] : 7u; }
uint32_t gs_condition(void) { if (gs_g.c) return 1u; return 0u; }
uint64_t gs_sizes(void) { uint64_t x = gs_g.n; return x + sizeof gs_g + sizeof gs_t2.in.a; }
uint32_t gs_device(uint32_t i) { gs_dev.regs[i & 3u] = i; gs_dev.plain = gs_dev.regs[(i + 1u) & 3u]; return gs_dev.plain; }

/* initialized globals keep their declared type */
uint64_t gs_wide_plus(void) { return gs_wide + 1u; }
uint64_t gs_wide_shift(void) { return gs_wide >> 1; }
int64_t gs_small_scaled(void) { return gs_small * 1000; }
uint64_t gs_init_member(void) { return gs_init.n + gs_init.c; }
uint64_t gs_init_sizes(void) { return sizeof gs_init + 100u * sizeof gs_wide; }
