/* SP-ONCE's pass counts for the program rail's lowering: fuzz_decoder_pack.c's harness (as committed) replayed on seed
 * files, so callgrind can count the CRC-32 passes inside bcir_decoder_lower_pack. The harness drives each input raw
 * and resealed and lowers an admitted pack twice in each (the lowering and its repeat): four lowerings a seed.
 * usage: replay_pack SEED... */
#include <stdint.h>
#include <stdio.h>
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
int main(int argc, char **argv) {
  static uint8_t buf[1 << 18];
  for (int i = 1; i < argc; i++) {
    FILE *f = fopen(argv[i], "rb");
    if (!f) return 2;
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    if (n == sizeof buf) return 2;
    LLVMFuzzerTestOneInput(buf, n);
  }
  return 0;
}
