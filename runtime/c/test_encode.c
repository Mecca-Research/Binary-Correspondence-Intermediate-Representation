/*===- test_encode.c - Python<->C parity harness for the StreamPack encoder ===
 *
 * Reads a StreamPack file (argv[1], produced by the Python encoder), re-serializes it
 * through the C encoder (bcir_sp_reencode), and writes the result to argv[2]. The Python
 * side (bcir/tests/test_c_encoder.py) asserts the re-encoded bytes are **byte-identical**
 * to the original -- i.e. the C writer reproduces the frozen ABI exactly (the encoder's
 * parity gate, the write-side analog of test_runtime.c). Uses libc (a host harness).
 *
 *   clang -std=c23 test_encode.c bcir_encode.c bcir_runtime.c -I . -o t && ./t in.bin out.bin
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>

#include "bcir_encode.h"

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <in-pack> <out-pack>\n", argv[0]); return 2; }
  FILE *fp = fopen(argv[1], "rb");
  if (!fp) { perror("fopen in"); return 2; }
  static uint8_t in[1 << 20];
  size_t in_len = fread(in, 1, sizeof in, fp);
  fclose(fp);

  static uint8_t out[1 << 20];
  size_t out_len = 0;
  bcir_status st = bcir_sp_reencode(in, in_len, out, sizeof out, &out_len);
  if (st != BCIR_OK) { printf("ERR %d\n", (int)st); return 1; }

  FILE *op = fopen(argv[2], "wb");
  if (!op) { perror("fopen out"); return 2; }
  if (fwrite(out, 1, out_len, op) != out_len) { fclose(op); return 2; }
  fclose(op);
  printf("OK in=%zu out=%zu\n", in_len, out_len);
  return 0;
}
