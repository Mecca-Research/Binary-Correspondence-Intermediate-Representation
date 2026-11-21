/* CSUB-2c: an array member of an array element used as a value -- `bcir_channel.c`'s `strcmp(channels[i].name,
 * channels[best].name)` and `bcir_verify.c`'s `fixed_string(f->qcasts[k].type, sizeof f->qcasts[k].type)` --
 * behaviour-checked on both rails. A member array of scalars of an element is the pointer to its first element,
 * `&a[i].m[0]`: passed, assigned, chosen by `?:`, compared, subtracted, stepped, dereferenced and sized; of a block
 * static's element, of a pointer parameter's and through a loaded pointer member. The storage is block statics, set
 * at each call: the harness runs the original and then the emit, and the escape corpus holds every local array
 * private. */
#include <stdint.h>
#include <stddef.h>

#define NAME 6

struct channel {
  uint32_t rank;
  char name[NAME];
  uint16_t caps[3];
};
struct qcast {
  uint32_t operand;
  char type[NAME];
};
struct func {
  uint32_t n_qcasts;
  struct qcast *qcasts;
};

static int order(const char *a, const char *b) {
  uint32_t j = 0;
  for (; a[j] && a[j] == b[j]; j++) {
  }
  return (int)(uint8_t)a[j] - (int)(uint8_t)b[j];
}
static int fixed(const char *s, uint32_t n) {
  for (uint32_t i = 0; i < n; i++)
    if (!s[i]) return 1;
  return 0;
}
static uint32_t caps3(const uint16_t *c) { return c[0] + c[1] * 3u + c[2] * 5u; }

/* `bcir_channel.c`'s pick: the best rank, ties broken by the name */
static uint32_t pick(const struct channel *channels, uint32_t n) {
  uint32_t best = 0;
  for (uint32_t i = 1; i < n; i++)
    if (channels[i].rank > channels[best].rank ||
        (channels[i].rank == channels[best].rank && order(channels[i].name, channels[best].name) < 0))
      best = i;
  return best * 100u + caps3(channels[best].caps);
}

/* `bcir_verify.c`'s check: each cast's type a fixed string, through the pointer the function record holds */
static uint32_t verify(const struct func *f) {
  uint32_t bad = 0;
  for (uint32_t k = 0; k < f->n_qcasts; k++)
    if (!fixed(f->qcasts[k].type, sizeof f->qcasts[k].type)) bad = bad * 2u + k + 1u;
  return bad;
}

/* a block static's element: assigned, chosen, compared, subtracted, stepped and dereferenced */
static uint32_t values(uint32_t x) {
  static struct channel tab[3];
  for (uint32_t i = 0; i < 3u; i++) {
    tab[i].rank = (x + i) % 3u;
    for (uint32_t j = 0; j < NAME; j++) tab[i].name[j] = (char)(j + 1u < NAME ? 'a' + (x + i + j) % 5u : 0);
    tab[i].caps[0] = (uint16_t)(x + i);
    tab[i].caps[1] = (uint16_t)(i * 7u);
    tab[i].caps[2] = (uint16_t)(x ^ i);
  }
  uint32_t k = x % 3u;
  char *n;
  n = tab[k].name;
  n[0] = 'z';
  const uint16_t *c = (x & 1u) ? tab[0].caps : tab[2].caps;
  uint16_t *s = tab[k].caps + 1;
  *s += 2u;
  uint32_t r = caps3(c) + (uint32_t)*tab[k].name + *(tab[(k + 1u) % 3u].caps + 2) * 3u;
  r += (tab[k].name == n) + (uint32_t)(tab[2].caps - tab[0].caps) + (uint32_t)sizeof tab[k].name * 7u;
  return r + pick(tab, 3u) * 11u;
}

uint32_t csub2_elemarrays(uint32_t x, uint32_t y) {
  static struct qcast casts[3];
  static struct func f;
  for (uint32_t k = 0; k < 3u; k++) {
    casts[k].operand = k;
    for (uint32_t j = 0; j < NAME; j++) casts[k].type[j] = (char)('t' + j);
  }
  casts[(x + y) % 3u].type[NAME - 2u] = 0;
  casts[y % 3u].type[NAME - 1u] = (char)(y & 1u ? 0 : 'q');
  f.n_qcasts = 3u;
  f.qcasts = casts;
  return verify(&f) * 13u + values(x ^ y);
}
