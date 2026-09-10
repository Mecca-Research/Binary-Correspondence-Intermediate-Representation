// What the frontend does to a function signature before LLVM sees it.
// Companion chapter: llvm-training/20-clang-frontend/05-abi-and-target-lowering.md
//
//   clang -O0 -S -emit-llvm -target x86_64-unknown-linux-gnu abi-boundary.c -o -
//   clang -O0 -S -emit-llvm -target aarch64-unknown-linux-gnu abi-boundary.c -o -
//
// The C signatures below are identical across targets. The IR signatures are
// not: argument classification is the frontend's job, and it is target law.

struct Small { int a; int b; };            // 8 bytes: SysV passes it in one register
struct Medium { int a; int b; int c; };    // 12 bytes: two registers
struct Large { double v[8]; };             // 64 bytes: memory, by reference

struct Small  make_small(int a, int b);
struct Medium make_medium(void);
struct Large  make_large(void);

int take_small(struct Small s)   { return s.a + s.b; }
int take_medium(struct Medium m) { return m.a + m.b + m.c; }
double take_large(struct Large l) { return l.v[0] + l.v[7]; }

struct Small  return_small(void)  { return make_small(1, 2); }
struct Large  return_large(void)  { return make_large(); }

// Narrow integers: whether the caller or the callee extends is ABI, not style.
char narrow(char c, short s, _Bool b) { return (char)(c + s + b); }

// Variadic: the frontend emits the va_list machinery, not LLVM.
int sum_varargs(int n, ...);
int call_varargs(void) { return sum_varargs(3, 1, 2, 3); }
