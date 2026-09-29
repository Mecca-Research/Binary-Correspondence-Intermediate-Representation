/*===- test_execution_plan.c - ExecutionPlanV1 C harness (parity + verdicts) ---===
 *
 * Drives the freestanding plan decoder (bcir_execution_plan.h, in bcir_runtime.c) over a
 * Python-encoded plan and prints every decoded record as text, so the Python side can
 * rebuild the plan from the C decode and re-encode it: the G11 gate `plan.abi.roundtrip`
 * (Python encode -> C decode -> Python re-encode, byte-identical). With `--pack` it also
 * prints the plan/pack binding verdict (bcir_ep_check_pack) and with `--live` the R11
 * verdict against a caller registry (bcir_ep_check_generation_vector), so the stale and
 * malformed gates run on the C rail from the same harness.
 *
 * Usage:
 *   test_execution_plan <plan-file> [--dump] [--pack <pack-file>]
 *                       [--live rid:map_gen:data_gen ...]
 *
 * Prints `status=<BCIR_*>` (bcir_ep_validate), `verify=<BCIR_*>` (bcir_ep_verify), the
 * header line, the records under --dump (strings hex-encoded), `pack=<BCIR_*>` and
 * `vector=<BCIR_*>` when requested, then `OK` when every verdict is BCIR_OK. The exit
 * status is 0 exactly when it printed OK.
 *
 * The harness uses libc (it is a test); the runtime stays freestanding.
 *===----------------------------------------------------------------------===*/
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_execution_plan.h"

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
    case BCIR_ERR_PLAN:        return "BCIR_ERR_PLAN";
    default:                   return "BCIR_ERR_UNKNOWN";
  }
}

static void hex(const char *s, uint16_t n) {
  for (uint16_t i = 0; i < n; i++) printf("%02x", (unsigned)(uint8_t)s[i]);
}

static int on_step(const bcir_ep_step_view *v, void *ctx) {
  (void)ctx;
  printf("step claim=%" PRIu64 " phase=%u candidate=", v->claim_id, v->phase_id);
  hex(v->candidate, v->candidate_len);
  printf(" lane=%u width=%u cost=%" PRId64 " stream=%u start=%" PRIu64 " duration=%" PRIu64 "\n",
         v->lane, v->width, v->cost, v->stream, v->start, v->duration);
  return 0;
}

static int on_lifetime(const bcir_ep_lifetime_view *v, void *ctx) {
  (void)ctx;
  printf("lifetime rid=%u bank=", v->rid);
  hex(v->bank, v->bank_len);
  printf(" offset=%" PRIu64 " size=%" PRIu64 " alignment=%u first=%u last=%u first_tick=%" PRIu64
         " last_tick=%" PRIu64 "\n",
         v->offset, v->size, v->alignment, v->first_phase, v->last_phase, v->first_tick,
         v->last_tick);
  return 0;
}

static int on_move(const bcir_ep_move_view *v, void *ctx) {
  (void)ctx;
  printf("move rid=%u src=", v->rid);
  hex(v->src_bank, v->src_len);
  printf(" dst=");
  hex(v->dst_bank, v->dst_len);
  printf(" offset=%" PRIu64 " size=%" PRIu64 " route=", v->offset, v->size);
  hex(v->route, v->route_len);
  printf(" kind=%u coherence=%u map_gen=%u data_gen=%u after=%" PRIu64 " before=%" PRIu64
         " claim=%" PRIu64 " version=%u flags=%u producer=%" PRIu64 " bits=%u cert=%" PRIu64 "\n",
         v->kind, v->coherence, v->map_gen, v->data_gen, v->after_claim, v->before_claim,
         v->claim, v->version, v->flags, v->producer, v->bits, v->cert);
  return 0;
}

static int on_generation(const bcir_generation_view *g, void *ctx) {
  (void)ctx;
  printf("gen rid=%u map_gen=%u data_gen=%u\n", g->rid, g->map_gen, g->data_gen);
  return 0;
}

static uint8_t *slurp(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
  long len = ftell(f);
  if (len < 0) { fclose(f); return 0; }
  rewind(f);
  uint8_t *buf = (uint8_t *)malloc(len ? (size_t)len : 1u);
  if (!buf || (len && fread(buf, 1, (size_t)len, f) != (size_t)len)) { free(buf); fclose(f); return 0; }
  fclose(f);
  *n = (size_t)len;
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <plan-file> [--dump] [--pack <pack-file>] [--live rid:map:data ...]\n",
            argv[0]);
    return 2;
  }
  int dump = 0;
  const char *pack_path = 0;
  static bcir_generation_view live[256];
  size_t n_live = 0;
  int have_live = 0;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--dump") == 0) dump = 1;
    else if (strcmp(argv[i], "--pack") == 0 && i + 1 < argc) pack_path = argv[++i];
    else if (strcmp(argv[i], "--live") == 0) {
      have_live = 1;
      while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0) {
        unsigned r, m, d;
        if (sscanf(argv[++i], "%u:%u:%u", &r, &m, &d) != 3 || n_live >= 256) {
          fprintf(stderr, "bad --live entry %s\n", argv[i]);
          return 2;
        }
        live[n_live].rid = r; live[n_live].map_gen = m; live[n_live].data_gen = d; n_live++;
      }
    } else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  size_t n = 0;
  uint8_t *buf = slurp(argv[1], &n);
  if (!buf) { fprintf(stderr, "read failed: %s\n", argv[1]); return 2; }

  int ok = 1;
  bcir_ep_header hdr;
  bcir_status st = bcir_ep_validate(buf, n, &hdr);
  printf("status=%s\n", status_name(st));
  if (st != BCIR_OK) ok = 0;
  if (ok) {
    st = bcir_ep_verify(buf, n);
    printf("verify=%s\n", status_name(st));
    if (st != BCIR_OK) ok = 0;
  }
  if (ok) {
    uint16_t sp_len = 0;
    const char *sp = bcir_ep_source_plan(buf, n, &sp_len);
    printf("header version=%u mode=%u liveness=%u streams=%u knee=%u makespan=%" PRIu64
           " module_hash=%" PRIu64 " target_hash=%" PRIu64
           " n_steps=%u n_lifetimes=%u n_moves=%u n_gens=%u source_plan=",
           hdr.version, hdr.mode, hdr.liveness, hdr.streams, hdr.knee, hdr.makespan,
           hdr.module_hash, hdr.target_hash, hdr.n_steps, hdr.n_lifetimes, hdr.n_moves,
           hdr.n_gens);
    hex(sp ? sp : "", sp ? sp_len : 0);
    printf("\n");
    if (dump) {
      uint64_t source_hash = 0, spec_hash = 0;
      if (bcir_ep_binding(buf, n, &source_hash, &spec_hash) != BCIR_OK) {
        printf("WALK_FAIL\n");
        ok = 0;
      }
      printf("binding source_hash=%" PRIu64 " spec_hash=%" PRIu64 "\n", source_hash, spec_hash);
      if (bcir_ep_for_each_step(buf, n, on_step, 0) != BCIR_OK ||
          bcir_ep_for_each_lifetime(buf, n, on_lifetime, 0) != BCIR_OK ||
          bcir_ep_for_each_move(buf, n, on_move, 0) != BCIR_OK ||
          bcir_ep_for_each_generation(buf, n, on_generation, 0) != BCIR_OK) {
        printf("WALK_FAIL\n");
        ok = 0;
      }
    }
  }
  if (pack_path) {
    size_t pn = 0;
    uint8_t *pack = slurp(pack_path, &pn);
    if (!pack) { fprintf(stderr, "read failed: %s\n", pack_path); free(buf); return 2; }
    st = bcir_ep_check_pack(buf, n, pack, pn);
    printf("pack=%s\n", status_name(st));
    if (st != BCIR_OK) ok = 0;
    free(pack);
  }
  if (have_live) {
    st = bcir_ep_check_generation_vector(buf, n, live, n_live);
    printf("vector=%s\n", status_name(st));
    if (st != BCIR_OK) ok = 0;
  }
  if (ok) printf("OK\n");
  free(buf);
  return ok ? 0 : 1;
}
