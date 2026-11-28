/* CSUB-2d: a row of a multi-dimensional array used as a value -- `bcir_channel.c`'s `strcmp(seen_keys[i], key)`,
 * `bcir_verify.c`'s `fixed_string(f->calls[k], BCIR_CIR_NAME)`, `bcir_ed25519.c`'s `fsub(a, p[1], p[0])` of a
 * `gf p[4]` parameter and `bcir_tensor.c`'s `exp_run(..., ys[v])` -- behaviour-checked on both rails. A row of scalars
 * is the pointer to its first element, `&m[k][0]`: passed to a call, assigned, initialized from, chosen by `?:`,
 * stepped, compared, subtracted, dereferenced and sized; of a block static, a global, a VLA, a `T m[][N]` parameter, a
 * typedef'd array's, a member array (through a value, a loaded pointer and an element of an array of structs) and the
 * row a pointer-to-array member points at. The address of a volatile array's element is a `volatile T *`. The storage
 * a call reads is block statics, set at each call: the harness runs the original and then the emit, and the escape
 * corpus holds every local array private. */
#include <stdint.h>
#include <stddef.h>

#define NAME 8
#define KEY 6

typedef int64_t gf[4];

struct func {
  uint32_t n;
  char (*calls)[NAME];
  int n_calls;
};
struct grid {
  uint16_t cell[3][4];
  uint32_t n;
};
struct slot {
  uint32_t id;
  uint8_t b[2][3];
};

static int same(const char *a, const char *b) {
  uint32_t j = 0;
  for (; a[j] && a[j] == b[j]; j++) {
  }
  return a[j] == b[j];
}
static uint32_t sum4(const uint16_t *r) { return r[0] + r[1] * 3u + r[2] * 5u + r[3] * 7u; }
static uint32_t sum3b(const uint8_t *r) { return r[0] + r[1] * 3u + r[2] * 5u; }
static void fadd(int64_t *o, const int64_t *a, const int64_t *b) {
  for (uint32_t i = 0; i < 4u; i++) o[i] = a[i] + b[i];
}
static void fsub(int64_t *o, const int64_t *a, const int64_t *b) {
  for (uint32_t i = 0; i < 4u; i++) o[i] = a[i] - b[i];
}
static float fsum(const float *v, uint32_t n) {
  float s = 0.0f;
  for (uint32_t i = 0; i < n; i++) s += v[i];
  return s;
}

/* `bcir_channel.c`'s duplicate keys: a key stored into its row a byte at a time, and each row compared */
static int seen(uint32_t x) {
  static char keys[4][KEY];
  static const char *const words[5] = {"id", "mode", "id", "rank", "mode"};
  uint32_t n = 0;
  int dup = 0;
  for (uint32_t w = 0; w < 5u; w++) {
    const char *key = words[(w + x) % 5u];
    for (uint32_t i = 0; i < n; i++)
      if (same(keys[i], key)) dup += (int)(i + 1u);
    if (n == sizeof keys / sizeof keys[0]) break;
    char *row = keys[n];
    uint32_t j = 0;
    for (; key[j] && j + 1u < KEY; j++) row[j] = key[j];
    row[j] = 0;
    n++;
  }
  return dup * 10 + (int)n;
}

/* `bcir_verify.c`'s call table: the row a pointer-to-array member points at, passed on */
static int fixed(const char *s, uint32_t n) {
  for (uint32_t i = 0; i < n; i++)
    if (!s[i]) return 1;
  return 0;
}
static uint32_t calls_ok(const struct func *f) {
  uint32_t r = 0;
  for (int k = 0; k < f->n_calls; k++) r = r * 2u + (uint32_t)fixed(f->calls[k], NAME);
  return r;
}

/* `bcir_ed25519.c`'s field arithmetic: a row of a `gf p[4]` parameter, and of a block static of `gf`s */
static void point_mix(gf p[4], gf q[4]) {
  gf a, b;
  fsub(a, p[1], p[0]);
  fadd(b, q[1], q[0]);
  fadd(p[2], a, b);
  fsub(q[3], p[3], q[2]);
}
static int64_t points(uint32_t x) {
  static gf p[4], q[4];
  for (uint32_t i = 0; i < 4u; i++)
    for (uint32_t j = 0; j < 4u; j++) {
      p[i][j] = (int64_t)(x + i * 4u + j);
      q[i][j] = (int64_t)(x ^ (i + j)) - 3;
    }
  point_mix(p, q);
  return p[2][1] + q[3][3] * 3 + p[2][3];
}

/* `bcir_tensor.c`'s variants: each a row of a float array */
static float variants(uint32_t x) {
  static float ys[3][5];
  for (uint32_t v = 0; v < 3u; v++)
    for (uint32_t i = 0; i < 5u; i++) ys[v][i] = (float)(x % 7u) + (float)(v * 5u + i) * 0.5f;
  float s = 0.0f;
  for (uint32_t v = 0; v < 3u; v++) s += fsum(ys[v], 5u) * (float)(v + 1u);
  return s;
}

static uint16_t gm[3][4];

/* a row assigned, initialized from, chosen, stepped, compared, subtracted, dereferenced and sized */
static uint32_t values(uint32_t x, uint32_t y) {
  static uint16_t m[3][4];
  for (uint32_t i = 0; i < 3u; i++)
    for (uint32_t j = 0; j < 4u; j++) {
      m[i][j] = (uint16_t)(x + i * 4u + j);
      gm[i][j] = (uint16_t)(y ^ (i * 3u + j));
    }
  uint32_t k = x % 3u;
  uint16_t *r;
  r = m[k];
  const uint16_t *c = (y & 1u) ? m[0] : gm[2];
  uint16_t *s = m[k] + 1;
  *s += 2u;
  uint32_t res = sum4(r) + sum4(c) * 3u + sum4(gm[(k + 1u) % 3u]) * 5u + *m[k] + *(m[(k + 2u) % 3u] + 3);
  res += (m[k] == r) + (m[0] != m[1]) * 2u + (uint32_t)(m[2] - m[0]) * 4u + (uint32_t)sizeof m[k];
  uint16_t *rows[2] = {m[k], gm[0]};
  return res + rows[0][1] + rows[1][2] * 7u;
}

/* a row of a VLA and of a `T m[][N]` parameter */
static uint32_t rowsum(uint16_t rows[][4], uint32_t k) { return sum4(rows[k]); }
static uint32_t vla(uint32_t x, uint32_t n) {
  uint16_t v[n + 1u][4];
  for (uint32_t i = 0; i <= n; i++)
    for (uint32_t j = 0; j < 4u; j++) v[i][j] = (uint16_t)(x + i + j * 3u);
  uint16_t *r = v[x % (n + 1u)];
  return r[0] + r[3] * 3u + *v[n] + rowsum(gm, x % 3u);
}

/* a row of a member array: of a struct value, through a loaded pointer and of an array of structs' element */
static uint32_t members(uint32_t x) {
  static struct grid g;
  static struct slot sl[2];
  struct grid *pg = &g;
  for (uint32_t i = 0; i < 3u; i++)
    for (uint32_t j = 0; j < 4u; j++) g.cell[i][j] = (uint16_t)(x * 2u + i + j);
  for (uint32_t i = 0; i < 2u; i++)
    for (uint32_t j = 0; j < 3u; j++) {
      sl[i].b[0][j] = (uint8_t)(x + i + j);
      sl[i].b[1][j] = (uint8_t)(x * 3u + i * j);
    }
  uint16_t *r = pg->cell[x % 3u];
  r[2] = (uint16_t)(r[2] + 9u);
  uint8_t *b = sl[x & 1u].b[1];
  b[0] ^= 0x5au;
  return sum4(g.cell[1]) + sum4(pg->cell[2]) * 3u + sum3b(sl[x & 1u].b[0]) * 5u + sum3b(b) * 7u + *pg->cell[0];
}

/* the address of a volatile array's element, and of a volatile member's */
struct dev {
  volatile uint16_t reg;
  volatile uint16_t fifo[4];
};
static uint32_t volatiles(uint32_t x) {
  static volatile uint16_t a[4];
  static struct dev d;
  volatile uint16_t *p = &a[x & 3u];
  *p = (uint16_t)(x + 5u);
  volatile uint16_t *q = &d.reg;
  *q = (uint16_t)(x * 3u);
  volatile uint16_t *f = &d.fifo[(x + 1u) & 3u];
  *f = 11u;
  return a[x & 3u] + d.reg * 3u + d.fifo[(x + 1u) & 3u] * 5u;
}

uint32_t csub2_rows(uint32_t x, uint32_t y) {
  static char names[3][NAME];
  static struct func f;
  for (uint32_t k = 0; k < 3u; k++)
    for (uint32_t j = 0; j < NAME; j++) names[k][j] = (char)((k == 1u && j == NAME - 1u) ? 'z' : (j < 3u ? 'a' + k : 0));
  names[(x % 2u) * 2u][NAME - 1u] = (char)('q' + (y & 1u));
  f.calls = names;
  f.n_calls = 3;
  uint32_t r = (uint32_t)seen(x) + calls_ok(&f) * 3u;
  r += (uint32_t)points(x) * 5u + (uint32_t)variants(y) * 7u;
  r += values(x, y) * 11u + vla(x, y % 4u) * 13u + members(x ^ y) * 17u + volatiles(y) * 19u;
  return r;
}
