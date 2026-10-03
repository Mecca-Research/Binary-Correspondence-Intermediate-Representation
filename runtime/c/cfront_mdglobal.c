/* Multi-dimensional subscripts and array-of-structs member arrays (CF-SMALL): a 2-D/3-D file-scope array
 * indexed in full is one row-major element of the whole object -- read, written, compound-assigned, stepped
 * and addressed; `a[i].m[j]` on an array of structs folds the element index with the member's, `i*K + j` in
 * the member's elements, for every access form (a read, `=`/`OP=`/`++`/`--` as values and as statements,
 * `&`), and a 2-D member takes its row-major index; `&m[i][j]` of a local, a VLA or a `T m[][N]` parameter
 * flattens every subscript. Every global the entry reads is reset first, so a run does not depend on the
 * last one. Both rails lower them the same way, and each emit returns what the original does. */
#include <stdint.h>

uint8_t md_g2[3][5];
uint32_t md_g3[2][3][4];
struct md_s { uint8_t c; uint64_t n; uint32_t a[4]; };
struct md_s md_aos[3];
struct md_t { uint16_t h; uint16_t m[2][3]; };
struct md_t md_mt[4];
struct md_u { uint8_t b[3]; uint8_t z; };
struct md_u md_nb[5];

static void md_reset(void) {
  for (uint32_t i = 0u; i < 3u; i++) {
    for (uint32_t j = 0u; j < 5u; j++) md_g2[i][j] = (uint8_t)(i * 5u + j);
  }
  for (uint32_t i = 0u; i < 2u; i++) {
    for (uint32_t j = 0u; j < 3u; j++) {
      for (uint32_t k = 0u; k < 4u; k++) md_g3[i][j][k] = i * 100u + j * 10u + k;
    }
  }
  for (uint32_t i = 0u; i < 3u; i++) {
    md_aos[i].c = (uint8_t)i;
    md_aos[i].n = i;
    for (uint32_t k = 0u; k < 4u; k++) md_aos[i].a[k] = i * 4u + k;
  }
  for (uint32_t i = 0u; i < 4u; i++) {
    md_mt[i].h = (uint16_t)i;
    for (uint32_t j = 0u; j < 2u; j++) {
      for (uint32_t k = 0u; k < 3u; k++) md_mt[i].m[j][k] = (uint16_t)(i + j + k);
    }
  }
  for (uint32_t i = 0u; i < 5u; i++) {
    md_nb[i].z = (uint8_t)i;
    for (uint32_t k = 0u; k < 3u; k++) md_nb[i].b[k] = (uint8_t)(240u + k);
  }
}
uint32_t md_read(uint32_t i, uint32_t j, uint32_t k) {   /* full subscripts of a 2-D and a 3-D global */
  return md_g2[i % 3u][j % 5u] + md_g2[2][4] * 3u + md_g3[i % 2u][j % 3u][k % 4u] * 5u
         + (uint32_t)(sizeof md_g3 + sizeof md_g3[1] + sizeof md_g3[1][2]);
}
uint32_t md_write(uint32_t i, uint32_t j, uint32_t k) {  /* =, OP=, ++ on a 3-D global's element */
  md_g3[i % 2u][j % 3u][k % 4u] = 5u;
  md_g3[i % 2u][j % 3u][k % 4u] += 7u;
  md_g3[1][2][3]++;
  uint32_t v = (md_g3[0][1][2] *= 3u);
  return md_g3[i % 2u][j % 3u][k % 4u] + md_g3[1][2][3] * 3u + v * 5u;
}
uint32_t md_address(uint32_t i, uint32_t j) {            /* &g[i][j] of a 2-D global */
  uint8_t *p = &md_g2[i % 3u][j % 5u];
  *p = (uint8_t)(*p + 9u);
  return md_g2[i % 3u][j % 5u] + *p * 3u;
}
uint32_t md_aos_read(uint32_t i, uint32_t j) {           /* a[i].m[j]: the folded element */
  return md_aos[i % 3u].a[j % 4u] + md_aos[2].a[3] * 3u + md_aos[i % 3u].c * 5u;
}
uint32_t md_aos_values(uint32_t i, uint32_t j) {         /* =, OP=, ++, -- as values */
  uint32_t x = (md_aos[i % 3u].a[j % 4u] = 5u);
  x += (md_aos[i % 3u].a[j % 4u] *= 3u);
  x += md_aos[i % 3u].a[j % 4u]++;
  x += --md_aos[i % 3u].a[j % 4u];
  return x + md_aos[i % 3u].a[j % 4u] * 7u;
}
uint32_t md_aos_stmts(uint32_t i, uint32_t j) {          /* =, OP=, ++ as statements */
  md_aos[i % 3u].a[j % 4u] = 11u;
  md_aos[i % 3u].a[j % 4u] += 2u;
  md_aos[i % 3u].a[j % 4u]++;
  return md_aos[i % 3u].a[j % 4u] + md_aos[(i + 1u) % 3u].a[j % 4u] * 3u;
}
uint32_t md_aos_address(uint32_t i, uint32_t j) {        /* &a[i].m[j] */
  uint32_t *p = &md_aos[i % 3u].a[j % 4u];
  *p += 4u;
  return md_aos[i % 3u].a[j % 4u] + *p * 3u;
}
uint32_t md_aos_rows(uint32_t i, uint32_t j, uint32_t k) {   /* a 2-D member array of an element */
  md_mt[i % 4u].m[j % 2u][k % 3u] += 1u;
  return md_mt[i % 4u].m[j % 2u][k % 3u] + md_mt[3].m[1][2] * 3u + md_mt[i % 4u].h * 5u;
}
uint32_t md_aos_narrow(uint32_t i, uint32_t j) {         /* a byte member element re-reads what it stored */
  uint32_t x = (md_nb[i % 5u].b[j % 3u] += 30u);
  uint32_t y = md_nb[i % 5u].b[j % 3u]++;
  return x + y * 3u + md_nb[i % 5u].b[j % 3u] * 5u + md_nb[i % 5u].z * 7u;
}
uint32_t md_local(uint32_t i, uint32_t j, uint32_t k) {  /* &m[i][j] of a 2-D and a 3-D local */
  uint8_t m[3][5] = {0};
  uint32_t c[2][3][4] = {0};
  uint8_t *p = &m[i % 3u][j % 5u];
  uint32_t *q = &c[i % 2u][j % 3u][k % 4u];
  *p = 9u;
  *q = 7u;
  return m[1][2] + c[1][2][3] * 3u + *p * 5u + *q * 7u;
}
uint32_t md_vla(uint32_t n, uint32_t i, uint32_t j) {    /* &v[i][j] of a 2-D VLA */
  uint32_t rows = n % 3u + 1u;
  uint8_t v[rows][5];
  for (uint32_t r = 0u; r < rows; r++) {
    for (uint32_t c = 0u; c < 5u; c++) v[r][c] = (uint8_t)(r * 5u + c);
  }
  uint8_t *p = &v[i % rows][j % 5u];
  *p = (uint8_t)(*p + 1u);
  return *p + v[rows - 1u][4] * 3u;
}
uint32_t md_param(uint8_t m[][5], uint32_t i, uint32_t j) {   /* &m[i][j] of a `T m[][N]` parameter */
  uint8_t *p = &m[i % 3u][j % 5u];
  return *p + m[2][4] * 3u;
}
uint32_t md_param_call(uint32_t i, uint32_t j) {
  uint8_t m[3][5] = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
  return md_param(m, i, j);
}
uint32_t md_entry(uint32_t i, uint32_t j, uint32_t k) {   /* one call per statement: each writes the globals
                                                          * the next reads, so their order is fixed */
  md_reset();
  uint32_t r = md_read(i, j, k);
  r += md_write(i, j, k) * 3u;
  r += md_address(i, j) * 5u;
  r += md_aos_read(i, j) * 7u;
  r += md_aos_values(i, j) * 11u;
  r += md_aos_stmts(i, j) * 13u;
  r += md_aos_address(i, j) * 17u;
  r += md_aos_rows(i, j, k) * 19u;
  r += md_aos_narrow(i, j) * 23u;
  r += md_local(i, j, k) * 29u;
  r += md_vla(i, j, k) * 31u;
  r += md_param_call(i, j) * 37u;
  return r;
}
