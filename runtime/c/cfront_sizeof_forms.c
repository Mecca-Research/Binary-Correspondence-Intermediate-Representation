/* CF-SIZEOF: `sizeof` measures its operand's own type, unevaluated (C11 6.5.3.4p2) -- an array is its whole
 * array, a row of a multi-dimensional one a row, a member array its member, never the pointer it decays to
 * anywhere else -- and its value is a `size_t` (p5). Both rails had folded a bare name's declared element
 * size or a flat 4 for most of these (`sizeof(la + 1)` and `sizeof ls.a` were 4 on the oracle, `sizeof gb`
 * 4 on the twin, whose `sizeof` bound only a primary, so `sizeof -x` lowered `4 - x`), and typed the result
 * a 32-bit unsigned, so `sizeof(uint32_t) - 8u` wrapped at 2^32. Each function returns the folded value, or
 * a size_t computation over it; the test runs the original and both emits and compares every one. */
#include <stdint.h>
#include <stddef.h>

struct sz_s { uint8_t c; uint64_t n; uint32_t a[4]; };
struct sz_t { uint16_t k; struct sz_s in; uint8_t z[3]; };
union sz_u { uint8_t b[5]; uint32_t w; };
struct sz_bf { uint32_t lo : 3; uint32_t hi : 29; uint64_t big : 40; uint8_t tiny : 2; };
struct sz_big { uint64_t w[2]; };
enum sz_e { SZ_A, SZ_B = 7 };

uint32_t sz_ga[10];
uint8_t sz_gb[3][5];
struct sz_s sz_gs;
struct sz_s sz_gsa[3];
uint16_t sz_gi[] = {1, 2, 3, [6] = 4};
uint64_t sz_gw = 5u;

static uint16_t sz_h(uint32_t v) { return (uint16_t)v; }
static struct sz_s sz_mk(void) { struct sz_s v = {0}; return v; }

/* locals: an array, a row, an element, a member array, a union, a nested member */
uint64_t sz_local_array(void) { uint32_t la[10]; return sizeof la; }
uint64_t sz_local_row(void) { uint16_t l2[4][6]; return sizeof l2[1] + 100u * sizeof l2; }
uint64_t sz_local_elem(void) { uint16_t l2[4][6]; return sizeof l2[1][2] + 100u * sizeof *l2 + 10000u * sizeof **l2; }
uint64_t sz_local_decays(void) { uint32_t la[10]; return sizeof(la + 1) + 100u * sizeof(&la) + 10000u * sizeof(*la); }
uint64_t sz_member_array(void) { struct sz_s ls; return sizeof ls.a + 100u * sizeof ls.a[1] + 10000u * sizeof ls; }
uint64_t sz_nested(void) { struct sz_t lt; return sizeof lt.in.a + 100u * sizeof lt.z + 10000u * sizeof lt; }
uint64_t sz_struct_array(void) { struct sz_s lsa[4]; return sizeof lsa + 1000u * sizeof lsa[1].a + 100000u * sizeof lsa->n; }
uint64_t sz_union(void) { union sz_u lu; return sizeof lu + 100u * sizeof lu.b; }
uint64_t sz_brace_sized(void) { uint32_t bi[] = {1, 2, 3}; return sizeof bi + bi[1]; }

/* parameters: a pointer, a decayed array, a row of a decayed 2-D array, a row pointer */
uint64_t sz_param_ptr(uint32_t *p, struct sz_s *sp) { return sizeof p + 100u * sizeof *p + 10000u * sizeof *sp + 1000000u * sizeof sp->a; }
uint64_t sz_param_array(uint32_t ap[8]) { return sizeof ap + 100u * sizeof ap[0]; }
uint64_t sz_param_2d(uint16_t m[4][6]) { return sizeof m[1] + 100u * sizeof m[1][2] + 10000u * sizeof *m; }
uint64_t sz_param_rowptr(uint16_t (*pr)[6]) { return sizeof pr + 100u * sizeof *pr + 10000u * sizeof pr[1][2]; }

/* file-scope objects: every dimension, a member array, an array of structs, an initializer-sized array */
uint64_t sz_global(void) { return sizeof sz_ga + 1000u * sizeof sz_gb + 100000u * sizeof sz_gb[1]; }
uint64_t sz_global_struct(void) { return sizeof sz_gs + 100u * sizeof sz_gs.a + 10000u * sizeof sz_gsa + 1000000u * sizeof sz_gsa[2].a; }
uint64_t sz_global_sized(void) { return sizeof sz_gi + 100u * sizeof sz_gw; }

/* type-names: a scalar, a pointer, arrays of them, a struct array */
uint64_t sz_types(void) { return sizeof(uint32_t[10]) + 1000u * sizeof(uint8_t *[5]) + 100000u * sizeof(struct sz_s[2]); }
uint64_t sz_types2(void) { return sizeof(uint32_t[3][4]) + 1000u * sizeof(uint64_t **) + 100000u * sizeof(enum sz_e); }
uint64_t sz_alignof(void) { return _Alignof(uint64_t) + 100u * _Alignof(struct sz_s) + 10000u * _Alignof(uint16_t[5]); }

/* expressions: promotions, comparisons, shifts, casts, a conditional, a comma, an assignment, an increment */
uint64_t sz_promote(uint8_t x8, int64_t x64) { return sizeof(x8 + 1) + 100u * sizeof(-x8) + 10000u * sizeof(-x64) + 1000000u * sizeof(x8 << 1); }
uint64_t sz_bare(int64_t x64, struct sz_s *sp) { return sizeof -x64 + 100u * sizeof *sp; }
uint64_t sz_compare(uint64_t x, uint32_t y) { return sizeof(x == 1u) + 100u * sizeof(!x) + 10000u * sizeof((uint8_t)y) + 1000000u * sizeof(x && y); }
uint64_t sz_conditional(uint8_t c, uint64_t w, uint32_t *p, uint32_t *q) { return sizeof(c ? c : w) + 100u * sizeof(c ? p : q) + 10000u * sizeof(c, w); }
uint64_t sz_assign(uint8_t x8, uint64_t w) { return sizeof(x8 = w) + 100u * sizeof(x8++) + 10000u * sizeof(++x8) + 1000000u * sizeof(x8 += 300); }
uint64_t sz_float(float f, double d) { return sizeof(f * 2.0f) + 100u * sizeof(f * 2.0) + 10000u * sizeof(d + 1) + 1000000u * sizeof 1.0f; }
uint64_t sz_literals(void) { return sizeof 'a' + 100u * sizeof("abc") + 10000u * sizeof("ab" "cde") + 1000000u * sizeof(1ull); }
uint64_t sz_calls(void) { return sizeof(sz_h(1u)) + 100u * sizeof(sz_mk()) + 10000u * sizeof(sz_mk().n); }
uint64_t sz_bitfield_ops(void) { struct sz_bf b; return sizeof(b.lo + 1) + 100u * sizeof(b.big + 1) + 10000u * sizeof(-b.tiny) + 1000000u * sizeof b; }
uint64_t sz_bitfield_keeps(void) { struct sz_bf b = {0}; return sizeof(b.tiny = 1) + 100u * sizeof(b.tiny++) + 10000u * sizeof(0, b.tiny); }
uint64_t sz_nested_sizeof(uint8_t x8) { return sizeof(sizeof x8) + 100u * sizeof(_Alignof(uint64_t)) + 10000u * sizeof(SZ_B); }
uint64_t sz_pointer_difference(uint32_t *p, uint32_t *q) { return sizeof(p - q) + 100u * sizeof(p + 1); }

/* sizeof is unevaluated: an index, an increment or a call in it never runs */
uint64_t sz_unevaluated(uint32_t i) { uint32_t la[10] = {0}; uint32_t j = i; uint64_t s = sizeof la[j++] + sizeof(sz_h(j++)); return s * 100u + j; }

/* a VLA's size is a runtime value; its element's is not */
uint64_t sz_vla(uint32_t n) { uint32_t v[n % 7u + 1u]; v[0] = 1u; return sizeof v + 1000u * sizeof v[0]; }

/* the value is a size_t: arithmetic on it is pointer-wide and unsigned */
int64_t sz_size_t_wraps(void) { return (int64_t)(sizeof(uint32_t) - 8u); }
uint64_t sz_size_t_scales(uint32_t n) { return sizeof(struct sz_big) * n; }
uint64_t sz_array_count(void) { uint32_t la[10]; return sizeof la / sizeof la[0]; }
