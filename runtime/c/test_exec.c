/*===- test_exec.c - Python<->C parity harness for the StreamPack executor ===
 *
 * Reads a StreamPack file (argv[1], produced by the Python encoder), runs the
 * deterministic executor, and prints the dispatch order + per-phase telemetry. The
 * Python side (bcir/tests/test_c_executor.py) runs bcir.gem.execute on the same module
 * and asserts the order + (scheduled, executed) agree -- the executor's parity gate (the
 * analog of test_runtime.c for the decoder). A counting kernel records the dispatch order
 * to prove the per-claim callback fires in the right sequence. Uses libc (a host harness).
 *
 *   clang -std=c23 test_exec.c bcir_exec.c bcir_runtime.c -I . -o t && ./t pack.bin
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>

#include "bcir_exec.h"

static uint64_t g_order[4096];
static size_t g_norder;

static int record(const bcir_exec_item *item, void *ctx) {
  (void)ctx;
  if (g_norder < sizeof g_order / sizeof g_order[0])
    g_order[g_norder++] = item->claim_id;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <pack-file>\n", argv[0]); return 2; }
  FILE *fp = fopen(argv[1], "rb");
  if (!fp) { perror("fopen"); return 2; }
  static uint8_t buf[1 << 20];
  size_t len = fread(buf, 1, sizeof buf, fp);
  fclose(fp);

  static bcir_exec_item scratch[4096];
  static bcir_phase_stat phases[256];
  bcir_exec_result res;
  bcir_status st = bcir_sp_execute(buf, len, scratch, 4096, phases, 256, record, NULL, &res);
  if (st != BCIR_OK) { printf("ERR %d\n", (int)st); return 1; }

  {
    bcir_exec_result failed = {(size_t)-1, (size_t)-1, (size_t)-1};
    bcir_phase_stat sentinel = {0xA5A5A5A5u, 0xA5A5A5A5u, 0xA5A5A5A5u};
    st = bcir_sp_execute(buf, len, scratch, 4096, &sentinel, 0, NULL, NULL, &failed);
    if (st != BCIR_ERR_NOSPACE || failed.executed || failed.n_phases ||
        failed.n_segments || sentinel.phase_id != 0xA5A5A5A5u ||
        sentinel.scheduled != 0xA5A5A5A5u || sentinel.executed != 0xA5A5A5A5u) {
      fprintf(stderr, "executor failure-state regression\n");
      return 3;
    }
  }

  /* The dispatch order, as recorded by the kernel callback (must equal gem.execute). */
  printf("order:");
  for (size_t i = 0; i < g_norder; i++) printf(" %llu", (unsigned long long)g_order[i]);
  printf("\n");
  /* Per-phase telemetry, in execution (first-appearance / topological) order. */
  for (size_t i = 0; i < res.n_phases; i++)
    printf("phase %u %u %u\n", phases[i].phase_id, phases[i].scheduled, phases[i].executed);
  printf("executed %zu\n", res.executed);
  return 0;
}
