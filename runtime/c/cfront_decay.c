/* CF-DECAY / CF-PTRDIFF / CF-DEREFIDX: an array used as a value is the address of its first element, and
 * the difference of two pointers is a `ptrdiff_t`. Both rails read a member array used as a value (`gs.a`,
 * `l.a`, `sp->a`) as a load of its first element: `return gs.a;` did not compile, and `(uintptr_t)gs.a`
 * returned that element's value. Both typed `la + 1` and a conditional over pointer arms as a 32-bit
 * integer (`int32_t t = la + 1;`, uncompilable), and `p - q` as one too -- the twin an UNSIGNED one, so
 * `p - q` of -2 returned 4294967294. And the twin lowered the index of `*(la + j++)` twice, reading
 * `la[j + 1]` and stepping `j` by two. Each function below runs against the original, on both emits.
 * Where an array's address leaves the value model (`dk_w32 - e`, `(uintptr_t)dk_w32`) the array is a file-scope
 * one: a local's would escape, rightly, and `escape.unproved` counts every local array of the corpus the escape
 * analysis does not prove private, so the test runs those local forms inline. Every function is defined for
 * every argument, as the G10 commute witness calls them with any. */
#include <stdint.h>

struct dk_s { uint8_t c; uint64_t n; uint32_t a[4]; };
struct dk_w { uint16_t h; struct dk_s in; };

struct dk_s dk_g;
uint32_t dk_w32[16];

int64_t dk_diff_u8(uint8_t *p, uint8_t *q) { return p - q; }
int64_t dk_diff_u32(uint32_t *p, uint32_t *q) { return p - q; }
int64_t dk_diff_array(void) { uint32_t *e = dk_w32 + 6; return dk_w32 - e; }
uint32_t dk_array_plus(uint32_t i) { uint32_t la[10] = {0}; la[3] = 7u; uint32_t *p = la + 1; return p[i % 5u]; }
uint32_t *dk_global_plus(uint32_t i) { return dk_w32 + (i % 16u); }
double dk_float_plus(void) { double ld[4] = {1.5, 2.5, 3.5, 4.5}; double *p = ld + 2; return *p; }
uintptr_t dk_array_as_integer(void) { return (uintptr_t)(dk_w32 + 1) - (uintptr_t)dk_w32; }
uint32_t *dk_select(uint32_t c) { return c ? dk_w32 : dk_w32 + 2; }
uint32_t dk_select_array(uint32_t c) { uint32_t la[4] = {1, 2, 3, 4}; uint32_t *p = c ? la : la + 2; return p[1]; }
double dk_select_float(uint32_t c) { double ld[4] = {1.5, 2.5, 3.5, 4.5}; double *p = c ? ld : ld + 3; return *p; }
uint32_t *dk_member_value(void) { return dk_g.a; }
uint32_t dk_member_local(uint32_t i) { struct dk_s l = {0}; l.a[2] = 9u; uint32_t *q = l.a; return q[i % 4u]; }
uint32_t dk_member_arrow(struct dk_s *sp) { uint32_t *q = sp->a; return q[1] + sp->a[1]; }
uint32_t dk_member_nested(void) { struct dk_w l = {0}; l.in.a[3] = 11u; uint32_t *q = l.in.a; return q[3]; }
uint32_t dk_member_select(uint32_t c) { uint32_t *p = c ? dk_g.a : dk_w32; return p[1]; }
uintptr_t dk_member_offset(void) { return (uintptr_t)dk_g.a - (uintptr_t)&dk_g; }
uint32_t dk_deref_index(uint32_t i) { uint32_t la[10] = {0}; la[3] = 7u; uint32_t j = i % 10u; uint32_t v = *(la + j++); return v * 100u + j; }
uint32_t dk_deref_index_value(uint32_t i) { uint32_t la[10] = {0}; la[3] = 7u; return *(la + (i % 10u)); }
