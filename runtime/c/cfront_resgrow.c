/* The twin's resource table grows while a pointer temporary is being made (the #sanitize witness): each
 * `rgN` declares N chained locals before `a + (k & 1u)`, so across N = 4..20 one of them makes the pointer
 * temporary at the exact count where `add_res` reallocates the table (its first growth, at 16). `tempptr`
 * read its source resource through a pointer into the old table after the reallocation -- a heap
 * use-after-free in the compiler itself, visible only to the sanitized twin, which
 * `tools/c/sanitize_cfront.sh` runs over every fixture. Both rails lower it the same way, and each emit
 * returns what the original does. */
#include <stdint.h>
uint32_t rg4(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v4;
}
uint32_t rg5(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v5;
}
uint32_t rg6(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v6;
}
uint32_t rg7(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v7;
}
uint32_t rg8(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v8;
}
uint32_t rg9(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v9;
}
uint32_t rg10(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t v10 = v9 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v10;
}
uint32_t rg11(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t v10 = v9 + 1u;
  uint32_t v11 = v10 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v11;
}
uint32_t rg12(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t v10 = v9 + 1u;
  uint32_t v11 = v10 + 1u;
  uint32_t v12 = v11 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v12;
}
uint32_t rg13(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t v10 = v9 + 1u;
  uint32_t v11 = v10 + 1u;
  uint32_t v12 = v11 + 1u;
  uint32_t v13 = v12 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v13;
}
uint32_t rg14(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t v10 = v9 + 1u;
  uint32_t v11 = v10 + 1u;
  uint32_t v12 = v11 + 1u;
  uint32_t v13 = v12 + 1u;
  uint32_t v14 = v13 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v14;
}
uint32_t rg15(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t v10 = v9 + 1u;
  uint32_t v11 = v10 + 1u;
  uint32_t v12 = v11 + 1u;
  uint32_t v13 = v12 + 1u;
  uint32_t v14 = v13 + 1u;
  uint32_t v15 = v14 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v15;
}
uint32_t rg16(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t v10 = v9 + 1u;
  uint32_t v11 = v10 + 1u;
  uint32_t v12 = v11 + 1u;
  uint32_t v13 = v12 + 1u;
  uint32_t v14 = v13 + 1u;
  uint32_t v15 = v14 + 1u;
  uint32_t v16 = v15 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v16;
}
uint32_t rg17(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t v10 = v9 + 1u;
  uint32_t v11 = v10 + 1u;
  uint32_t v12 = v11 + 1u;
  uint32_t v13 = v12 + 1u;
  uint32_t v14 = v13 + 1u;
  uint32_t v15 = v14 + 1u;
  uint32_t v16 = v15 + 1u;
  uint32_t v17 = v16 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v17;
}
uint32_t rg18(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t v10 = v9 + 1u;
  uint32_t v11 = v10 + 1u;
  uint32_t v12 = v11 + 1u;
  uint32_t v13 = v12 + 1u;
  uint32_t v14 = v13 + 1u;
  uint32_t v15 = v14 + 1u;
  uint32_t v16 = v15 + 1u;
  uint32_t v17 = v16 + 1u;
  uint32_t v18 = v17 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v18;
}
uint32_t rg19(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t v10 = v9 + 1u;
  uint32_t v11 = v10 + 1u;
  uint32_t v12 = v11 + 1u;
  uint32_t v13 = v12 + 1u;
  uint32_t v14 = v13 + 1u;
  uint32_t v15 = v14 + 1u;
  uint32_t v16 = v15 + 1u;
  uint32_t v17 = v16 + 1u;
  uint32_t v18 = v17 + 1u;
  uint32_t v19 = v18 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v19;
}
uint32_t rg20(uint32_t k) {
  uint32_t a[2] = {k, k + 7u};
  uint32_t v1 = k;
  uint32_t v2 = v1 + 1u;
  uint32_t v3 = v2 + 1u;
  uint32_t v4 = v3 + 1u;
  uint32_t v5 = v4 + 1u;
  uint32_t v6 = v5 + 1u;
  uint32_t v7 = v6 + 1u;
  uint32_t v8 = v7 + 1u;
  uint32_t v9 = v8 + 1u;
  uint32_t v10 = v9 + 1u;
  uint32_t v11 = v10 + 1u;
  uint32_t v12 = v11 + 1u;
  uint32_t v13 = v12 + 1u;
  uint32_t v14 = v13 + 1u;
  uint32_t v15 = v14 + 1u;
  uint32_t v16 = v15 + 1u;
  uint32_t v17 = v16 + 1u;
  uint32_t v18 = v17 + 1u;
  uint32_t v19 = v18 + 1u;
  uint32_t v20 = v19 + 1u;
  uint32_t *q = a + (k & 1u);
  return *q + v20;
}
uint32_t rg_entry(uint32_t s) {
  return rg4(s) + rg5(s) + rg6(s) + rg7(s) + rg8(s) + rg9(s) + rg10(s) + rg11(s) + rg12(s) +
         rg13(s) + rg14(s) + rg15(s) + rg16(s) + rg17(s) + rg18(s) + rg19(s) + rg20(s);
}
