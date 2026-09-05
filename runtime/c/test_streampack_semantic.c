/*===- test_streampack_semantic.c - C semantic trust-boundary harness -----===
 *
 * Drives the freestanding semantic verifier over a (CRC-fixed) StreamPack file and
 * prints the verdict, so tools/c/check_streampack_semantic.sh can assert each
 * CRC-VALID-but-semantically-corrupt pack is now REJECTED on the C rail (R10/R11 +
 * the lane/width/dispatch range gate), where it used to execute silently.
 *
 * Usage:
 *   test_streampack_semantic <pack-file> [expected_map_gen expected_data_gen]
 *                            [--live rid:map_gen:data_gen ...]
 *
 * Prints one line:  status=<code> name=<BCIR_*>  and a second line `exec=<code>`
 * showing what the executor's checked entry point does with the same pack. With a
 * `--live` registry it also prints `vector=<BCIR_*>` (bcir_sp_check_generation_vector,
 * the per-resource R11 of StreamPack v4) and `exec_vector=<BCIR_*>` (the vector-checked
 * executor), and the exit status follows the vector verdict.
 *
 * The harness uses libc (it is a test); the runtime stays freestanding.
 *===----------------------------------------------------------------------===*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_exec.h"
#include "bcir_verify.h"

static const char *status_name(bcir_status s) {
  switch (s) {
    case BCIR_OK:              return "BCIR_OK";
    case BCIR_ERR_TRUNCATED:   return "BCIR_ERR_TRUNCATED";
    case BCIR_ERR_MAGIC:       return "BCIR_ERR_MAGIC";
    case BCIR_ERR_VERSION:     return "BCIR_ERR_VERSION";
    case BCIR_ERR_CRC:         return "BCIR_ERR_CRC";
    case BCIR_ERR_NOSPACE:     return "BCIR_ERR_NOSPACE";
    case BCIR_ERR_LANE:        return "BCIR_ERR_LANE";
    case BCIR_ERR_WIDTH:       return "BCIR_ERR_WIDTH";
    case BCIR_ERR_DISPATCH:    return "BCIR_ERR_DISPATCH";
    case BCIR_ERR_PROVENANCE:  return "BCIR_ERR_PROVENANCE";
    case BCIR_ERR_STALE:       return "BCIR_ERR_STALE";
    case BCIR_ERR_OVERFLOW:    return "BCIR_ERR_OVERFLOW";
    case BCIR_ERR_TRAILING:    return "BCIR_ERR_TRAILING";
    case BCIR_ERR_RESERVED:    return "BCIR_ERR_RESERVED";
    case BCIR_ERR_UTF8:        return "BCIR_ERR_UTF8";
    case BCIR_ERR_GENERATION:  return "BCIR_ERR_GENERATION";
    default:                   return "BCIR_ERR_UNKNOWN";
  }
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <pack-file> [map_gen data_gen]\n", argv[0]); return 2; }
  uint32_t emap = 0xFFFFFFFFu, edata = 0xFFFFFFFFu;
  static bcir_generation_view live[64];
  size_t n_live = 0;
  int have_live = 0;
  {
    int i = 2;
    if (argc >= 4 && strcmp(argv[2], "--live") != 0) {
      emap = (uint32_t)strtoul(argv[2], NULL, 10); edata = (uint32_t)strtoul(argv[3], NULL, 10);
      i = 4;
    }
    if (i < argc && strcmp(argv[i], "--live") == 0) {
      have_live = 1;
      for (i = i + 1; i < argc && n_live < 64; i++) {
        unsigned rid = 0, mg = 0, dg = 0;
        if (sscanf(argv[i], "%u:%u:%u", &rid, &mg, &dg) != 3) { fprintf(stderr, "bad --live entry %s\n", argv[i]); return 2; }
        live[n_live].rid = rid; live[n_live].map_gen = mg; live[n_live].data_gen = dg; n_live++;
      }
    }
  }

  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror("fopen"); return 2; }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *buf = (uint8_t *)malloc((size_t)n);
  if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read failed\n"); return 2; }
  fclose(f);

  /* The CRC must still validate (these are CRC-FIXED corruptions): if the CRC fails the
   * corruption was not crafted to defeat byte-identity, so report it explicitly. */
  bcir_streampack_header hdr;
  bcir_status v = bcir_sp_validate(buf, (size_t)n, &hdr);
  printf("crc=%s\n", status_name(v));

  bcir_status sem = bcir_sp_verify_semantic(buf, (size_t)n, emap, edata);
  printf("status=%d name=%s\n", (int)sem, status_name(sem));

  /* The public graph-verifier entry point must enforce the same artifact-intrinsic
   * R10/R11 rules. It deliberately cannot diagnose a stale live generation because
   * that registry state is not part of its API. */
  char diag[128];
  int verified = v == BCIR_OK &&
      bcir_verify_pack(buf, (size_t)n, hdr.n_segments, diag, sizeof diag);
  printf("verify_pack=%s\n", verified ? "BCIR_OK" : "BCIR_ERR");

  /* What the fully-checked executor does with the same pack (must refuse a corrupt one). */
  static bcir_exec_item scratch[8192];
  static bcir_phase_stat phases[256];
  bcir_exec_result res;
  bcir_status ex = bcir_sp_execute_checked(buf, (size_t)n, emap, edata,
                                           scratch, 8192, phases, 256, NULL, NULL, &res);
  printf("exec=%s\n", status_name(ex));

  /* StreamPack v4: the per-resource R11 against the caller's live registry, and the
   * vector-checked executor -- the verdict a driver holding its registry gets. */
  bcir_status vec = BCIR_OK;
  if (have_live) {
    vec = bcir_sp_check_generation_vector(buf, (size_t)n, live, n_live);
    printf("vector=%s\n", status_name(vec));
    bcir_status exv = bcir_sp_execute_checked_vector(buf, (size_t)n, live, n_live,
                                                     scratch, 8192, phases, 256, NULL, NULL, &res);
    printf("exec_vector=%s\n", status_name(exv));
  }

  free(buf);
  if (have_live) return vec == BCIR_OK ? 0 : 1;
  return sem == BCIR_OK ? 0 : 1;
}
