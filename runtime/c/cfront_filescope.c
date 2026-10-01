/* File-scope objects as C types them and the emits name them (CF-CHARELEM, CF-STRELEM, CF-AOS2D, CF-DEREFSUM,
 * CF-PARENSTR, CF-LINKEMIT). The twin had typed a plain `char` element `int8_t` -- a signed char, where AArch64's
 * `char` is unsigned -- and a string literal's element `int`; both rails refused `gm[i][j].x` of a file-scope array
 * of structs; the twin read `*(p + i - j)` as `p[i - j]`, the index computed unsigned -- a silent miscompile -- and
 * refused `char s[] = ("abc");`, which the oracle lowered; both emits read a `const` global into a temp of their own,
 * unqualified, type, which Clang refuses. Each form here lowers to one claim graph on the four targets and runs as
 * the original under Clang and GCC, with a signed and with an unsigned `char`. Every function leaves each object it
 * writes as it found it, or writes it before reading it, so the original and either emit read the same values. */
#include <stdint.h>

struct fs_pt { uint32_t x, y; };
union fs_un { uint32_t w; uint8_t b[4]; };
struct fs_in { uint16_t k; uint16_t m; };
struct fs_ot { uint32_t a; struct fs_in in; };
struct fs_rec { uint32_t x; uint8_t a[4]; };
typedef struct { uint32_t a; uint16_t b; } fs_row_t;

static char fs_cs[4] = {1, 2, (char)200, 4};
static signed char fs_sc[2] = {-3, 5};
static unsigned char fs_uc[2] = {250u, 7u};
static struct fs_pt fs_gm[2][3] = {{{1u, 2u}, {3u, 4u}}, {{5u, 6u}}};
static struct fs_pt fs_g3[2][2][2];
static union fs_un fs_gu[2][2];
static struct fs_ot fs_gn[2][2];
static struct fs_rec fs_gr[2][3];
static uint32_t fs_arr[8] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
static uint32_t fs_six[6] = {1u, 2u, 3u, 4u, 5u, 6u};
static uint32_t *fs_gp[2][3] = {{&fs_six[0], &fs_six[1], &fs_six[2]}, {&fs_six[3], &fs_six[4], &fs_six[5]}};
static const char *fs_gs[2][2] = {{"ab", "cd"}, {"ef", "gh"}};
static char fs_ps[] = ("abc");
static const uint32_t fs_k = 7u;
static const uint32_t fs_k3[3] = {1u, 2u, 3u};
static const uint32_t *fs_kq = &fs_k3[1];
static const char *fs_cp = "abc";
static const char *const fs_names[] = {"ab", "cde", "f"};
static const uint32_t fs_k22[2][2] = {{1u, 2u}, {3u, 4u}};
static const uint16_t fs_k222[2][2][2] = {{{1u, 2u}, {3u, 4u}}, {{5u, 6u}, {7u, 8u}}};
static const struct fs_pt fs_cpt[2][2] = {{{1u, 2u}, {3u, 4u}}, {{5u, 6u}, {7u, 8u}}};
static const fs_row_t fs_rows[3] = {{1u, 2u}, {3u, 4u}, {5u, 6u}};
static const double fs_kd[3] = {1.5, 2.5, 3.5};
static char fs_buf[4] = {'a', 'b', 'c', 0};
static char *const fs_bp = fs_buf;

static uint32_t fs_sum4(const uint32_t *p) { return p[0] + p[1] + p[2]; }
static uint32_t fs_sumpt(struct fs_pt v) { return v.x + v.y * 3u; }

uint32_t fs_chars(uint32_t s) {          /* a plain `char` element is a `char`, whatever its sign on the target */
  char c = fs_cs[s & 3u];
  char d = (char)s;
  uint32_t r = (uint32_t)(int)c + (uint32_t)(int)d * 3u + (uint32_t)(c < 0) * 5u + (uint32_t)(fs_cs[2] > 100) * 7u;
  r += (uint32_t)(int)fs_sc[s & 1u] * 11u + (uint32_t)fs_uc[s & 1u] * 13u + (uint32_t)(int)fs_cs[(s >> 2) & 3u] * 17u;
  r += (uint32_t)(int)(char)(s >> 8) * 31u + _Generic((char)s, char: 37u, default: 41u);
  return r + _Generic(fs_cs[0], char: 19u, default: 23u) + (uint32_t)sizeof(fs_cs[1] + 0) * 29u;
}
uint32_t fs_strelem(uint32_t s) {        /* a string literal's element has the literal's element type */
  uint32_t r = (uint32_t)(int)*"\xf0" + (uint32_t)(int)*("ab\xf1" + (s & 2u)) * 3u;
  r += (uint32_t)(int)*(2 + "ab\xf1") * 5u + (uint32_t)(int)*("abc" + 1 + (s & 1u)) * 7u;
  r += (uint32_t)(int)*(("abc") + (s & 1u)) * 11u + (uint32_t)*(L"ab" + (s & 1u)) * 13u;
  r += _Generic(u"x"[0], unsigned short: 1u, default: 2u) * 17u + _Generic(U"x"[0], unsigned: 3u, default: 5u) * 19u;
  r += _Generic("x"[0], char: 7u, default: 11u) * 23u + (uint32_t)sizeof(L"x"[0]) * 29u;
  return r + (uint32_t)sizeof(u"ab"[0]) * 31u + (uint32_t)sizeof(u"a" "b") * 37u + (uint32_t)sizeof("a" U"b") * 41u;
}
uint32_t fs_aos2d(uint32_t s) {          /* `gm[i][j].x` of a file-scope array of structs: read, stored, stepped */
  fs_gm[0][1].x = 7u;
  fs_gm[1][0].y = 9u;
  fs_gm[1][2].y = 1u;
  fs_gm[s & 1u][s % 3u].x = s;
  fs_gm[1][2].y += s;
  fs_gm[0][1].x++;
  ++fs_gm[1][0].y;
  return fs_gm[s & 1u][s % 3u].x + fs_gm[1][2].y * 3u + fs_gm[0][1].x * 5u + fs_gm[1][0].y * 7u;
}
uint32_t fs_aos3d(uint32_t s) {          /* three dimensions, a union, a nested member, a member array */
  fs_g3[1][0][1].x = 4u;
  fs_g3[s & 1u][1][(s >> 1) & 1u].y = s;
  fs_g3[1][0][1].x += 2u;
  fs_gu[s & 1u][1].w = s;
  fs_gn[0][1].in.m = 5u;
  fs_gn[1][s & 1u].in.k = (uint16_t)s;
  fs_gn[0][1].in.m += 3u;
  fs_gr[1][2].a[3] = (uint8_t)s;
  fs_gr[s & 1u][0].a[s & 3u] = 1u;
  fs_gr[s & 1u][0].a[s & 3u] += 2u;
  uint32_t r = fs_g3[s & 1u][1][(s >> 1) & 1u].y + fs_g3[1][0][1].x * 3u + fs_gu[s & 1u][1].b[0] * 5u;
  r += fs_gn[1][s & 1u].in.k * 7u + fs_gn[0][1].in.m * 11u + fs_gr[1][2].a[3] * 13u;
  return r + fs_gr[s & 1u][0].a[s & 3u] * 17u;
}
uint32_t fs_aoscopy(uint32_t s) {        /* an element copied, its address taken, passed by value; a static of them */
  static struct fs_pt sm[2][2];
  struct fs_pt v = fs_gm[s & 1u][1];
  struct fs_pt keep = fs_gm[1][2];
  fs_gm[1][2] = v;
  uint32_t r = v.x + v.y * 3u + fs_gm[1][2].x * 5u + fs_gm[0][0].y * 7u;
  fs_gm[1][2] = keep;
  struct fs_pt *p = &fs_gm[1][s % 3u];
  uint32_t old = p->x;
  p->x = s;
  r += (uint32_t)sizeof fs_gm[1] * 11u + (uint32_t)sizeof fs_gm[1][2] * 13u + fs_gm[1][s % 3u].x * 17u;
  p->x = old;
  sm[1][1].y = 3u;
  sm[s & 1u][1].x = s;
  r += sm[s & 1u][1].x + sm[1][1].y * 19u;
  const struct fs_pt *q = &fs_gm[s & 1u][0];
  r += q[1].x + q[2].y * 23u + (uint32_t)(&fs_gm[1][2] - &fs_gm[0][0]) * 29u;
  return r + fs_sumpt(fs_gm[1][0]) * 31u + fs_sumpt(fs_gm[0][s % 3u]) * 37u;
}
uint32_t fs_ptr2d(uint32_t s) {          /* two dimensions of pointers: the element read, and read through */
  uint32_t *lp[2][3] = {{&fs_six[0], &fs_six[1], &fs_six[2]}, {&fs_six[3], &fs_six[4], &fs_six[5]}};
  uint32_t *p = fs_gp[1][s % 3u];
  uint32_t r = *fs_gp[s & 1u][s % 3u] + *fs_gp[1][2] * 3u + *p * 5u + fs_gp[s & 1u][2][0] * 7u;
  r += *lp[s & 1u][s % 3u] * 11u + *lp[1][2] * 13u;
  const char *t = fs_gs[s & 1u][1];
  return r + (uint32_t)fs_gs[s & 1u][(s >> 1) & 1u][1] * 17u + (uint32_t)fs_gs[1][0][0] * 19u + (uint32_t)t[1] * 23u;
}
uint32_t fs_derefsum(uint32_t s) {       /* `*(p + i - j)` is the element `p + i - j` points at, never `p[i - j]` */
  uint32_t *p = &fs_arr[4];
  int k = -1;
  uint32_t r = *(p + (s & 1u) - 1u) + *(p - 1u + (s & 1u)) * 3u + *(p + k) * 5u + *(p + (int)(s & 1u) - 2) * 7u;
  *(p + (s & 1u) - 2u) = s;
  r += fs_arr[2] * 11u + fs_arr[3] * 13u;
  fs_arr[2] = 3u;
  fs_arr[3] = 4u;
  *(p + (s & 1u) - 1u) += 5u;
  ++*(p + (s & 1u) - 1u);
  r += fs_arr[3] * 17u + fs_arr[4] * 19u;
  fs_arr[3] = 4u;
  fs_arr[4] = 5u;
  uint32_t x = (*(p + (s & 1u) - 1u) *= 3u);
  r += x + fs_arr[3] * 23u + fs_arr[4] * 29u;
  fs_arr[3] = 4u;
  fs_arr[4] = 5u;
  uint32_t *q = &fs_arr[2];
  return r + *(q + 1 + (s & 1u)) * 31u + *(q + (s & 3u) * 1u) * 37u + *(q + (s & 1u) + (s & 2u)) * 41u;
}
uint32_t fs_parenstr(uint32_t s) {       /* `char s[] = ("abc");`: the parentheses are redundant (C11 6.7.9p14) */
  char ls[] = ("xy");
  static char ss[] = ("pqr");
  uint32_t r = (uint32_t)fs_ps[s % 3u] + (uint32_t)ls[s & 1u] * 3u + (uint32_t)ss[s % 3u] * 5u;
  return r + (uint32_t)sizeof fs_ps * 7u + (uint32_t)sizeof ls * 11u + (uint32_t)sizeof ss * 13u;
}
uint32_t fs_consts(uint32_t s) {         /* a `const` global as the emit's temps meet it: through a cast */
  const uint32_t *p = fs_kq;
  const uint32_t *pk = &fs_k;
  const char *const *pn = fs_names;
  const struct fs_pt *q = &fs_cpt[s & 1u][1];
  const fs_row_t *w = &fs_rows[s % 3u];
  struct fs_pt v = fs_cpt[1][s & 1u];
  char *const *pb = &fs_bp;
  uint32_t r = p[s & 1u] + *fs_kq * 3u + *pk * 5u + fs_k * 7u + fs_k3[s % 3u] * 11u + fs_sum4(fs_k3) * 13u;
  r += (uint32_t)fs_cp[s & 1u] * 17u + (uint32_t)*fs_cp * 19u + (uint32_t)fs_names[s % 3u][0] * 23u;
  r += (uint32_t)pn[1][2] * 29u + fs_k22[s & 1u][1] * 31u + fs_k22[1][0] * 37u;
  r += (uint32_t)fs_k222[s & 1u][(s >> 1) & 1u][1] * 41u + (uint32_t)fs_k222[1][0][s & 1u] * 43u;
  r += q->x + fs_cpt[s & 1u][0].y * 47u + v.x * 53u + v.y * 59u + w->a * 61u + fs_rows[2].b * 67u;
  r += (uint32_t)(fs_kd[s % 3u] * 2.0) * 71u + (uint32_t)fs_bp[s & 1u] * 73u + (uint32_t)(*pb)[2] * 79u;
  const char *old = fs_cp;
  fs_cp = (s & 1u) ? "xy" : "zw";
  r += (uint32_t)fs_cp[1] * 83u;
  fs_cp = old;
  return r;
}
uint32_t fs_entry(uint32_t s) {
  return fs_chars(s) + fs_strelem(s) * 3u + fs_aos2d(s) * 5u + fs_aos3d(s) * 7u + fs_aoscopy(s) * 11u
         + fs_ptr2d(s) * 13u + fs_derefsum(s) * 17u + fs_parenstr(s) * 19u + fs_consts(s) * 23u;
}
