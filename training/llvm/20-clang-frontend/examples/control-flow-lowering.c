// How C control flow becomes basic blocks, and where the frontend already
// decides something the optimizer will be blamed for.
// Companion chapter: training/llvm/20-clang-frontend/03-c-lowering-rules.md
//
//   clang -O0 -S -emit-llvm control-flow-lowering.c -o -
//   clang -O1 -S -emit-llvm control-flow-lowering.c -o -   # phi nodes appear

int max_of(int a, int b) { return a > b ? a : b; }

// && and || are sequence points: the frontend must emit branches, never a
// bitwise `and`, because the right operand may not be evaluated at all.
int guarded_load(const int *p, int n) {
  if (p != 0 && n > 0)
    return p[0];
  return -1;
}

int loop_sum(const int *p, int n) {
  int acc = 0;
  for (int i = 0; i < n; ++i)
    acc += p[i];
  return acc;
}

int dispatch(int state) {
  switch (state) {
  case 0:  return 10;
  case 1:  return 20;
  case 2:  return 30;
  case 17: return 40;
  default: return -1;
  }
}

// `volatile` is a frontend promise carried into IR as an instruction flag; the
// optimizer honours it because the flag is there, not because of the C type.
int read_twice(volatile int *p) { return *p + *p; }
