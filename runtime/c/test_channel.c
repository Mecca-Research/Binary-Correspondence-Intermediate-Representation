/*===- test_channel.c - drive the C channel router for the Python<->C parity gate ----------===
 * Loads channel.json files (argv[1..]) and, for each claim read from stdin ("op stride_int"),
 * prints the required capabilities, the routed channel, and per-channel eligibility -- the line
 * bcir/tests/test_c_channel.py checks against bcir/channels on the same claims + channels.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_channel.h"

/* the capability bitmask as sorted names (matches Python's sorted(set) ordering). */
static void cap_names(uint32_t m, char *out, size_t n) {
  struct { uint32_t b; const char *nm; } C[] = {
    {BCIR_CAP_DATA_PARALLEL, "data_parallel"}, {BCIR_CAP_GATHER, "gather"},
    {BCIR_CAP_MATMUL, "matmul"}, {BCIR_CAP_REDUCE, "reduce"},
    {BCIR_CAP_SCALAR_STREAM, "scalar_stream"}, {BCIR_CAP_STREAM_UNIT, "stream_unit"},
    {BCIR_CAP_TILE, "tile"}, {BCIR_CAP_UNIVERSAL, "universal"}, {0, NULL}};
  size_t w = 0; out[0] = 0; int first = 1;
  for (int i = 0; C[i].nm; i++)
    if (m & C[i].b) { w += (size_t)snprintf(out + w, n - w, "%s%s", first ? "" : ",", C[i].nm); first = 0; }
}

int main(int argc, char **argv) {
  static bcir_channel chs[16]; int nch = 0;
  for (int i = 1; i < argc && nch < 16; i++) {
    FILE *f = fopen(argv[i], "rb");
    if (!f) { fprintf(stderr, "open %s\n", argv[i]); return 2; }
    static char buf[1 << 16];
    size_t n = fread(buf, 1, sizeof buf - 1, f); buf[n] = 0; fclose(f);
    char diag[128];
    if (bcir_channel_parse(buf, n, &chs[nch], diag, sizeof diag)) {
      fprintf(stderr, "parse %s: %s\n", argv[i], diag); return 1;
    }
    nch++;
  }
  char line[256];
  while (fgets(line, sizeof line, stdin)) {
    char op[64]; int sc = 0;
    if (sscanf(line, "%63s %d", op, &sc) < 1) continue;
    bcir_claim cl; memset(&cl, 0, sizeof cl);
    snprintf(cl.op, sizeof cl.op, "%s", op); cl.stride = (bcir_stride)sc;
    char caps[256]; cap_names(bcir_claim_required_caps(&cl), caps, sizeof caps);
    int r = bcir_channel_route(&cl, chs, nch);
    printf("%s|%d|caps=%s|route=%s|suits=", op, sc, caps, r >= 0 ? chs[r].name : "NONE");
    for (int i = 0; i < nch; i++)
      printf("%s%s=%d", i ? ";" : "", chs[i].name, bcir_channel_suits(&cl, &chs[i]));
    printf("\n");
  }
  return 0;
}
