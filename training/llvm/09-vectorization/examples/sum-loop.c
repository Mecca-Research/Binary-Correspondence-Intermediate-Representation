// A small loop-vectorization diagnostic example.
//
// Try:
//   clang -O3 -Rpass=loop-vectorize sum-loop.c -c -o /tmp/sum-loop.o
//   clang -O3 -Rpass-missed=loop-vectorize sum-loop.c -c -o /tmp/sum-loop.o

int sum_loop(const int *a, int n) {
  int sum = 0;

  for (int i = 0; i < n; ++i)
    sum += a[i];

  return sum;
}

void add_arrays(const int *restrict a,
                const int *restrict b,
                int *restrict c,
                int n) {
  for (int i = 0; i < n; ++i)
    c[i] = a[i] + b[i];
}
