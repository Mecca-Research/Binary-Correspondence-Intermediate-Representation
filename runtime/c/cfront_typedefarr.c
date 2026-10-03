/* A typedef'd array type (C11 6.7.8, CF-SMALL): `typedef T row_t[N]` names the whole array type, so a
 * declarator's own dimensions are OUTER to it -- `row_t rw[2]` is two rows, `typedef row_t mat_t[3]` a
 * 3 x N matrix -- and the type keeps its shape wherever it is spelled: a local, a struct member, a file-scope
 * object, `sizeof`, a compound literal's type-name, a typedef of the typedef. (A pointer to such a type has no
 * spelling on either rail and is refused.) Both rails lay it out the same way, and each emit returns what
 * the original does. */
#include <stdint.h>

typedef uint16_t row_t[4];
typedef row_t mat_t[3];
typedef uint8_t tag_t[5];
struct td_rec { tag_t tag; row_t vals; uint8_t n; };
row_t td_grow;                                      /* a file-scope object of the typedef'd array */
mat_t td_gmat;

uint32_t td_local(uint32_t i) {                     /* a local row: four elements */
  row_t r = {1, 2, (uint16_t)i, 4};
  return r[i % 4u] + r[2] * 3u + (uint32_t)sizeof r * 100u;
}
uint32_t td_rows(uint32_t i) {                      /* the declarator's dimension is outer: two rows */
  row_t rw[2] = {{1, 2, 3, 4}, {5, 6, 7, (uint16_t)i}};
  return rw[i % 2u][3] + rw[1][0] * 3u + (uint32_t)sizeof rw * 100u;
}
uint32_t td_matrix(uint32_t i) {                    /* a typedef of the typedef: 3 x 4 */
  mat_t m = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, (uint16_t)i}};
  m[i % 3u][i % 4u] = 50u;
  return m[2][3] + m[1][2] * 3u + m[0][0] * 5u + (uint32_t)sizeof(mat_t) * 100u;
}
uint32_t td_member(uint32_t i) {                    /* a struct member of the typedef'd type */
  struct td_rec r = {{1, 2, 3, 4, 5}, {6, 7, 8, (uint16_t)i}, 9};
  r.vals[i % 4u] += 2u;
  return r.tag[i % 5u] + r.vals[3] * 3u + r.vals[i % 4u] * 5u + r.n * 7u + (uint32_t)sizeof(struct td_rec) * 100u;
}
uint32_t td_global(uint32_t i) {                    /* file-scope objects of the typedef'd types */
  for (uint32_t k = 0u; k < 4u; k++) td_grow[k] = (uint16_t)(k * 3u + i);
  for (uint32_t k = 0u; k < 3u; k++) td_gmat[k][k] = (uint16_t)(k + i);
  return td_grow[i % 4u] + td_grow[3] * 3u + td_gmat[i % 3u][i % 3u] * 5u + (uint32_t)sizeof td_gmat * 100u;
}
uint32_t td_literal(uint32_t i) {                   /* the type-name of a compound literal */
  return (row_t){1, 2, (uint16_t)i, 4}[2] + (row_t){7}[i % 4u] * 3u;
}
uint32_t td_entry(uint32_t i) {
  return td_local(i) + td_rows(i) * 3u + td_matrix(i) * 5u + td_member(i) * 7u + td_global(i) * 11u
         + td_literal(i) * 13u;
}
