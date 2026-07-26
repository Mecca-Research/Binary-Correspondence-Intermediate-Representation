/* Harness for the DER -> native StreamPack fast path.
 *
 *   test_asn1_streampack <projection.der> [expected-native.bin]
 *
 * Reconstructs the native artifact from the DER projection and prints its length and
 * CRC. With `expected-native.bin` it additionally asserts BYTE IDENTITY against the
 * artifact the Python encoder produced -- law A3 on the C rail, with no Python in the
 * reconstruction path. Prints "OK <len>" on success. */
#include <stdio.h>
#include <stdlib.h>

#include "bcir_asn1_streampack.h"

static unsigned char *slurp(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return 0; }
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
  long size = ftell(f);
  if (size < 0) { fclose(f); return 0; }
  rewind(f);
  unsigned char *buf = (unsigned char *)malloc((size_t)size ? (size_t)size : 1);
  if (!buf) { fclose(f); return 0; }
  if (size && fread(buf, 1, (size_t)size, f) != (size_t)size) {
    fclose(f); free(buf); return 0;
  }
  fclose(f);
  *len = (size_t)size;
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <der> [expected-native]\n", argv[0]); return 2; }
  size_t der_len = 0;
  unsigned char *der = slurp(argv[1], &der_len);
  if (!der) return 2;

  size_t cap = bcir_asn1_streampack_bound(der_len);
  unsigned char *out = (unsigned char *)malloc(cap);
  if (!out) { free(der); return 2; }

  size_t out_len = 0;
  bcir_status st = bcir_asn1_to_streampack(der, der_len, out, cap, &out_len);
  if (st != BCIR_OK) {
    printf("FAIL status=%d\n", (int)st);
    free(der); free(out);
    return 1;
  }

  if (argc >= 3) {
    size_t want_len = 0;
    unsigned char *want = slurp(argv[2], &want_len);
    if (!want) { free(der); free(out); return 2; }
    if (want_len != out_len) {
      printf("FAIL length %zu != expected %zu\n", out_len, want_len);
      free(der); free(out); free(want);
      return 1;
    }
    for (size_t i = 0; i < out_len; i++) {
      if (out[i] != want[i]) {
        printf("FAIL byte %zu: got %02x want %02x\n", i, out[i], want[i]);
        free(der); free(out); free(want);
        return 1;
      }
    }
    free(want);
  }

  printf("OK %zu\n", out_len);
  free(der);
  free(out);
  return 0;
}
