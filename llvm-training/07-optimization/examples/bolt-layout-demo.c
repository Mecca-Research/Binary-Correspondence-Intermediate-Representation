// BOLT layout demo fixture.
//
// Build with relocations so a post-link optimizer can rewrite the executable:
//   clang -O2 -fno-omit-frame-pointer -Wl,--emit-relocs \
//     llvm-training/07-optimization/examples/bolt-layout-demo.c \
//     -o /tmp/bolt-layout-demo
//
// Expected symbols for layout inspection:
//   - main: dispatches a biased loop and should remain the entry point.
//   - hot_path: the frequently executed fast case.
//   - cold_path: the rare error-ish case that BOLT may move away from hot code.
//   - checksum: helper called by both paths.
//
// Expected blocks inside hot_path:
//   - entry/loop body for the common arithmetic trace.
//   - rare adjustment guarded by (x & 127) == 0; this block is intentionally
//     uncommon so BOLT has a recognizable cold basic-block candidate.
//
// The program is deterministic and needs no arguments, which makes it suitable
// for shell-based smoke checks and for collecting a small perf profile.

#include <stdint.h>
#include <stdio.h>

__attribute__((noinline)) static int checksum(int x) {
  int mix = x ^ (x << 7);
  mix ^= (uint32_t)mix >> 9;
  return mix & 0x3ff;
}

__attribute__((noinline)) static int hot_path(int x) {
  int acc = x;
  for (int i = 0; i < 16; ++i) {
    acc = (acc * 33) ^ checksum(acc + i);
  }

  if ((x & 127) == 0) {
    acc -= checksum(x + 4096);
  }

  return acc;
}

__attribute__((noinline)) static int cold_path(int x) {
  int acc = x;
  for (int i = 0; i < 4; ++i) {
    acc += checksum(acc - i) * 3;
  }
  return acc ^ 0x5a5a;
}

int main(void) {
  int total = 0;
  for (int i = 0; i < 20000; ++i) {
    if ((i % 97) == 0) {
      total += cold_path(i);
    } else {
      total += hot_path(i);
    }
  }

  printf("%d\n", total);
  return total == 0;
}
