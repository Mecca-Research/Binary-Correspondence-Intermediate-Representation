// How the frontend lays out aggregates, and what that becomes in IR.
// Companion chapter: llvm-training/20-clang-frontend/03-c-lowering-rules.md
//
//   clang -O0 -S -emit-llvm -target x86_64-unknown-linux-gnu struct-layout.c -o -
//
// Layout is an ABI decision the frontend has already made by the time LLVM sees
// anything. What reaches IR is the *result*: a struct kind (`{...}` vs packed
// `<{...}>`) whose natural padding follows from the module's data layout, with
// explicit padding fields materialized only where natural alignment cannot
// express the ABI's answer.

#include <stddef.h>

struct Padded {
  char tag;      // offset 0
  int  value;    // offset 4 -- three bytes of padding precede it
  char flag;     // offset 8
};               // sizeof == 12, alignof == 4

struct Packed {
  char tag;
  int  value;
  char flag;
} __attribute__((packed));  // sizeof == 6

struct Bits {
  unsigned kind  : 3;
  unsigned count : 12;
  unsigned wide  : 17;
};

union Word {
  unsigned u;
  float    f;
  unsigned char bytes[4];
};

int padded_value(const struct Padded *p) { return p->value; }
int packed_value(const struct Packed *p) { return p->value; }

unsigned bits_count(const struct Bits *b) { return b->count; }
void bits_set_count(struct Bits *b, unsigned n) { b->count = n; }

unsigned union_bits_of_float(float f) {
  union Word w;
  w.f = f;
  return w.u;
}

int array_element(const int a[static 8], int i) { return a[i]; }

size_t padded_size(void)     { return sizeof(struct Padded); }
size_t padded_offset(void)   { return offsetof(struct Padded, value); }
size_t packed_size(void)     { return sizeof(struct Packed); }
