/*===- bcir_asn1_bench.c - the native ASN.1 decode microbench protocol ------===
 *
 * J6 left one thing open, and it is the thing everything downstream waits on: there is no
 * way to produce a `measured` cost table, so `select_certified` refuses every timing
 * objective and phase H's decode-latency claim stays unsupportable. This is the harness
 * that produces one.
 *
 * WHAT IT MEASURES, AND WHY THAT IS THE HONEST CHOICE. Each candidate is timed on the work
 * a C peer actually does at a trust boundary: walk the octets it was handed and decide
 * whether they are well formed. That is `bcir_asn1_validate_der` for DER, the three-stage
 * bounded pass for JER, `bcir_asn1_validate` for BER, and a tag walk for XER. It is NOT a
 * schema-directed decode into typed values, because the C rail does not have one for every
 * candidate -- and timing a full decode against a structural scan would compare unlike work
 * and call the difference an encoding cost.
 *
 * WHAT IT REFUSES TO MEASURE. PER has no entry here and cannot have one: X.691 7.2 says a
 * PER encoding is not self-delimiting, so there IS no schema-free structural pass to time.
 * OER has none either, for the simpler reason that no C implementation exists yet. Both
 * absences are reported to the driver rather than filled in, because a table row invented
 * from a Python timing is exactly what J6's refusal exists to prevent.
 *
 * THE PROTOCOL, which is the part that makes the numbers comparable:
 *
 *   1. ONE corpus, identical octets for every round. A benchmark that regenerated its input
 *      would be timing the generator.
 *   2. Warmup rounds, discarded. The first pass over a cold buffer measures the memory
 *      system, not the parser.
 *   3. INTERLEAVED round-robin, not all rounds of A then all of B. A CPU that drifts in
 *      frequency or temperature during a run penalizes whichever candidate happened to run
 *      last; interleaving spreads that drift evenly across all of them, so it becomes noise
 *      in every sample rather than a bias in one. This is the single most important line in
 *      the file.
 *   4. Per-round MEDIAN of many iterations. `timespec_get` has coarse granularity on some
 *      hosts, so a single iteration can quantize to zero; the median of a batch is stable
 *      and needs no calibration of the clock itself.
 *   5. Every round's figure is emitted. The driver computes the interval from the order
 *      statistics -- this program does no statistics at all, so a change in how the interval
 *      is derived never requires re-running the measurement.
 *
 * Hosted, not freestanding: it needs a clock and stdio. It is a MEASUREMENT tool and never
 * a runtime component, which is why it lives in `hosted_tool` and links nothing from the
 * decode path but the decoders themselves.
 *
 * Line protocol, stdin:
 *
 *   rounds <warmup> <rounds> <iterations>
 *   case <label> <op> <hex>        op is der | ber | jer | xer
 *   run
 *
 * Output:
 *
 *   sample <label> <op> <round> <ns>     one per (case, round) after warmup
 *   unsupported <op> <reason>            an op this build cannot measure
 *   done <cases>
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bcir_asn1.h"
#include "bcir_emit.h"
#include "bcir_jer.h"
#include "bcir_xer.h"

#define MAX_CASES 32
#define MAX_BYTES (1 << 16)
#define MAX_LINE (MAX_BYTES * 2 + 64)
#define MAX_ROUNDS 256
#define MAX_DEPTH 64
#define SCRATCH (1 << 16)

static uint64_t now_ns(void) {
  /* ISO C11 timespec_get, matching bcir_microbench.c: no POSIX feature macros, and the
   * same clock the rest of the repository's measurement uses. */
  struct timespec t = {0, 0};
  if (timespec_get(&t, TIME_UTC) != TIME_UTC || t.tv_sec < 0 || t.tv_nsec < 0) return 0;
  return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

static int hex_nibble(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static long unhex(const char *text, unsigned char *out, size_t cap) {
  size_t n = 0;
  while (*text != '\0' && *text != '\n' && *text != '\r') {
    int hi = hex_nibble((unsigned char)text[0]);
    int lo = text[1] ? hex_nibble((unsigned char)text[1]) : -1;
    if (hi < 0 || lo < 0 || n >= cap) return -1;
    out[n++] = (unsigned char)((hi << 4) | lo);
    text += 2;
  }
  return (long)n;
}

#define MAX_PLAN_TEXT (1 << 14)
#define MAX_PLAN_NODES 256
#define MAX_PLAN_MEMBERS 256
/* Sized by the number of CONSTRAINED nodes, not by the node count: a constraint carries an
 * alphabet buffer and is two orders of magnitude larger than a node, while almost every
 * schema in the corpus has none. */
#define MAX_PLAN_CONSTRAINTS 32
#define MAX_PLAN_ENUMS 64
#define EMIT_SCRATCH 4096
#define EMIT_OUT (1 << 17)

typedef struct bench_case {
  char label[64];
  char op[8];
  unsigned char data[MAX_BYTES];
  size_t len;
  /* An ENCODE case carries a descriptor as well as a value; `encode` selects the arm. The
   * two live in one struct so the interleaved round-robin below can mix decode and encode
   * cases in a single run -- which matters, because comparing an encode number measured in
   * one process against a decode number measured in another reintroduces exactly the drift
   * the interleaving exists to remove. */
  int encode;
  bcir_emit_rules rules;
  bcir_emit_plan plan;
  bcir_emit_node nodes[MAX_PLAN_NODES];
  bcir_emit_member members[MAX_PLAN_MEMBERS];
  bcir_emit_constraint constraints[MAX_PLAN_CONSTRAINTS];
  bcir_emit_enum_item enums[MAX_PLAN_ENUMS];
} bench_case;

/* The work one candidate's peer does on the octets it was handed. Returns a value derived
 * from the result so the optimizer cannot delete the call -- a benchmark whose body is
 * dead code measures nothing, and at -O2 that is exactly what would happen. */
static uint64_t do_work(const bench_case *c) {
  uint64_t sink = 0;
  if (c->encode) {
    /* The plan-driven encoder (E2). Every candidate goes through the same descriptor and
     * the same neutral value stream, so what is timed is the ENCODING and not an adapter. */
    static uint8_t out[EMIT_OUT];
    static uint32_t scratch[EMIT_SCRATCH];
    size_t written = 0;
    bcir_emit_diag diag;
    sink += (uint64_t)bcir_emit(&c->plan, c->rules, c->data, c->len, out, sizeof(out),
                                &written, scratch, EMIT_SCRATCH, 32, &diag);
    sink += written;
    return sink;
  }
  if (strcmp(c->op, "der") == 0) {
    sink += (uint64_t)bcir_asn1_validate_der(c->data, c->len, 64);
  } else if (strcmp(c->op, "ber") == 0) {
    sink += (uint64_t)bcir_asn1_validate(c->data, c->len, 64);
  } else if (strcmp(c->op, "jer") == 0) {
    static bcir_jer_level stack[MAX_DEPTH];
    static uint8_t scratch[SCRATCH];
    bcir_jer_limits limits;
    bcir_jer_diag diag;
    uint64_t nodes = 0;
    bcir_jer_default_limits(&limits);
    /* The three stages in 4.2's order -- the same work `decode_bounded` does, so the
     * number is comparable to what a JER trust boundary actually pays. */
    sink += (uint64_t)bcir_jer_scan(c->data, c->len, &limits, stack, MAX_DEPTH, &nodes,
                                    &diag);
    sink += nodes;
    sink += (uint64_t)bcir_jer_validate_utf8(c->data, c->len, &diag);
    sink += (uint64_t)bcir_jer_parse(c->data, c->len, &limits, stack, MAX_DEPTH, scratch,
                                     sizeof(scratch), 0, 0, &diag);
  } else if (strcmp(c->op, "xer") == 0) {
    size_t pos = 0;
    while (pos < c->len) {
      bcir_xer_tag tag;
      pos = bcir_xer_skip_space((const char *)c->data, c->len, pos);
      if (pos >= c->len) break;
      if (c->data[pos] != '<') { pos++; continue; }
      if (bcir_xer_scan_tag((const char *)c->data, c->len, pos, &tag) != BCIR_XER_OK) break;
      if (tag.end <= pos) break;
      sink += tag.name_len;
      pos = tag.end;
    }
  }
  return sink;
}

static int cmp_u64(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return x < y ? -1 : (x > y ? 1 : 0);
}

int main(void) {
  static char line[MAX_LINE];
  static char hex[MAX_LINE];
  static bench_case cases[MAX_CASES];
  static uint64_t rounds_ns[MAX_CASES][MAX_ROUNDS];
  static uint64_t batch[4096];
  size_t n_cases = 0;
  long warmup = 2, rounds = 11, iterations = 64;
  uint64_t sink = 0;

  while (fgets(line, (int)sizeof(line), stdin) != NULL) {
    char op[32];
    if (sscanf(line, "%31s", op) != 1) continue;

    if (strcmp(op, "rounds") == 0) {
      if (sscanf(line, "%31s %ld %ld %ld", op, &warmup, &rounds, &iterations) != 4 ||
          warmup < 0 || rounds < 1 || rounds > MAX_ROUNDS ||
          iterations < 1 || (size_t)iterations > sizeof(batch) / sizeof(batch[0])) {
        printf("unsupported rounds out-of-range\n");
        return 2;
      }
      continue;
    }

    if (strcmp(op, "case") == 0) {
      bench_case *c;
      long len;
      if (n_cases >= MAX_CASES) {
        printf("unsupported case too-many\n");
        return 2;
      }
      c = &cases[n_cases];
      if (sscanf(line, "%31s %63s %7s %s", op, c->label, c->op, hex) != 4) {
        printf("unsupported case malformed\n");
        return 2;
      }
      if (strcmp(c->op, "der") != 0 && strcmp(c->op, "ber") != 0 &&
          strcmp(c->op, "jer") != 0 && strcmp(c->op, "xer") != 0) {
        /* Named rather than silently skipped: the driver must be able to tell "this build
         * cannot measure it" from "this build measured it as zero". */
        printf("unsupported %s no-native-decoder\n", c->op);
        return 2;
      }
      len = unhex(hex, c->data, sizeof(c->data));
      if (len < 0) {
        printf("unsupported case bad-hex\n");
        return 2;
      }
      c->len = (size_t)len;
      n_cases++;
      continue;
    }

    if (strcmp(op, "encase") == 0) {
      /* encase <label> <rules> <plan-hex> <stream-hex> */
      static unsigned char plan_text[MAX_PLAN_TEXT];
      static char plan_hex[MAX_LINE];
      bench_case *c;
      long plan_len, value_len;
      bcir_emit_diag diag;
      bcir_emit_tables tables;

      if (n_cases >= MAX_CASES) { printf("unsupported case too-many\n"); return 2; }
      c = &cases[n_cases];
      if (sscanf(line, "%31s %63s %7s %s %s", op, c->label, c->op, plan_hex, hex) != 5) {
        printf("unsupported encase malformed\n");
        return 2;
      }
      if (strcmp(c->op, "der") == 0) c->rules = BCIR_EMIT_DER;
      else if (strcmp(c->op, "ber") == 0) c->rules = BCIR_EMIT_BER;
      else if (strcmp(c->op, "jer") == 0) c->rules = BCIR_EMIT_JER;
      else if (strcmp(c->op, "coer") == 0) c->rules = BCIR_EMIT_COER;
      else { printf("unsupported %s no-native-encoder\n", c->op); return 2; }

      plan_len = unhex(plan_hex, plan_text, sizeof(plan_text));
      value_len = unhex(hex, c->data, sizeof(c->data));
      if (plan_len < 0 || value_len < 0) { printf("unsupported encase bad-hex\n"); return 2; }
      tables.nodes = c->nodes;             tables.node_cap = MAX_PLAN_NODES;
      tables.members = c->members;         tables.member_cap = MAX_PLAN_MEMBERS;
      tables.constraints = c->constraints; tables.constraint_cap = MAX_PLAN_CONSTRAINTS;
      tables.enums = c->enums;             tables.enum_cap = MAX_PLAN_ENUMS;
      if (bcir_emit_parse_plan((const char *)plan_text, (size_t)plan_len, &tables, &c->plan,
                               &diag) != BCIR_EMIT_OK) {
        printf("unsupported encase bad-plan\n");
        return 2;
      }
      c->len = (size_t)value_len;
      c->encode = 1;
      n_cases++;
      continue;
    }

    if (strcmp(op, "run") == 0) break;
  }

  if (n_cases == 0) {
    printf("done 0\n");
    return 0;
  }

  /* Warmup, over every case, before any sample is kept. The first pass touches cold
   * buffers and cold branch predictors; keeping it would measure the memory system. */
  for (long w = 0; w < warmup; w++)
    for (size_t i = 0; i < n_cases; i++)
      for (long it = 0; it < iterations; it++) sink += do_work(&cases[i]);

  /* INTERLEAVED: the outer loop is the round, the inner is the case. A CPU whose frequency
   * or temperature drifts during the run therefore affects every candidate in the same
   * round equally, instead of biasing whichever one was scheduled last. */
  for (long r = 0; r < rounds; r++) {
    for (size_t i = 0; i < n_cases; i++) {
      for (long it = 0; it < iterations; it++) {
        uint64_t t0 = now_ns();
        sink += do_work(&cases[i]);
        batch[it] = now_ns() - t0;
      }
      qsort(batch, (size_t)iterations, sizeof(batch[0]), cmp_u64);
      /* The median of the batch, not the mean: one preempted iteration must not move the
       * round's figure, and the clock's granularity quantizes short iterations to zero. */
      rounds_ns[i][r] = batch[iterations / 2];
    }
  }

  for (size_t i = 0; i < n_cases; i++)
    for (long r = 0; r < rounds; r++)
      printf("sample %s %s %ld %llu\n", cases[i].label, cases[i].op, r,
             (unsigned long long)rounds_ns[i][r]);
  printf("done %lu\n", (unsigned long)n_cases);
  /* Consume the sink so no call above is dead. */
  if (sink == 0xFFFFFFFFFFFFFFFFull) printf("sink %llu\n", (unsigned long long)sink);
  return 0;
}
