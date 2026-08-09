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
 * WHAT IT REFUSES TO MEASURE. PER has no DECODE entry here and cannot have one: X.691 7.2
 * says a PER encoding is not self-delimiting, so there IS no schema-free structural pass to
 * time. That is a law rather than a gap, and it does not bind the ENCODE side -- an encoder
 * is handed the type either way, which is why PER has four encode rows and no decode row.
 * The absence is reported to the driver rather than filled in, because a table row invented
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
 *   4. Per-round MEDIAN of timed GROUPS. `timespec_get` has coarse granularity on some
 *      hosts -- 52 ns on a Snapdragon 8 Gen 3, whose userspace clock is the 19.2 MHz ARM
 *      architectural timer -- so timing one iteration at a time rounds every sample to a
 *      tick, and a median of quantized samples stays quantized. Each span therefore covers
 *      a GROUP of iterations and is divided by the group size, which spreads the clock's
 *      error across the group; the median is taken over the groups, so a preempted span
 *      still moves one group rather than the round, and needs no calibration of the clock
 *      itself.
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
 *   encase <label> <op> <plan-hex> <hex>   op adds coer | {c,b}per-{a,u}
 *   run
 *
 * Output:
 *
 *   sample <label> <op> <round> <ns>     one per (case, round) after warmup
 *   unsupported <op> <reason>            an op this build cannot measure
 *   done <cases>
 *===----------------------------------------------------------------------===*/
/* Before any system header, so glibc exposes `syscall` for the counter probe below.
 *
 * The alternative -- declaring `syscall` and `ioctl` locally -- is what this file did first,
 * and it broke under Termux: glibc types ioctl's request as `unsigned long` and bionic types
 * it as `int`, so a local prototype matches one libc and collides with the other. Letting each
 * libc declare its own functions is the only version that is right on both, and a benchmark
 * that does not build on the target is a benchmark that cannot measure it.
 *
 * This does not change how the clock is read: `now_ns` still uses ISO C11 `timespec_get`
 * rather than POSIX `clock_gettime`, matching the rest of the repository's measurement. */
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bcir_asn1.h"
#include "bcir_emit.h"
#include "bcir_jer.h"
#include "bcir_oer.h"
#include "bcir_per_plan.h"
#include "bcir_xer.h"

#define MAX_CASES 32
#define MAX_BYTES (1 << 16)
#define MAX_LINE (MAX_BYTES * 2 + 64)

/* The scanf field width for a MAX_LINE token buffer, spelled as a literal because the
 * preprocessor cannot stringify an arithmetic expression into one. Every %s below carries it.
 * An unbounded %s is safe only while its buffer is as large as `line`, which is an invariant
 * nothing enforced -- `fields_hex` was a tenth the size and a plan of about a hundred members
 * wrote past its end. A width states the bound at the call instead of assuming it. */
#define MAX_LINE_SCAN "131135"
_Static_assert(MAX_LINE == 131136, "MAX_LINE_SCAN must remain MAX_LINE - 1");
#define MAX_ROUNDS 256
#define MAX_DEPTH 64
#define SCRATCH (1 << 16)

/* --- target hardware counters ------------------------------------------------------------
 *
 * WHY A CYCLE COUNT AND NOT JUST A CLOCK. Two reasons, and the first one is now measured
 * rather than hypothetical. A wall-clock figure is only as fine as the clock: on a Snapdragon
 * 8 Gen 3 userspace reads the 19.2 MHz architectural timer, so the finest thing it can say is
 * 52 ns, and a 104 ns cost is "two ticks". A cycle counter runs at core frequency and resolves
 * far below that. Second, a cycle count is frequency-INVARIANT: DVFS and thermal ramp move
 * nanoseconds and leave cycles alone, which is exactly the drift the interleaving above is
 * fighting.
 *
 * WHY IT IS OPTIONAL AND SAYS SO. `perf_event_open` needs a PMU the kernel is willing to
 * expose. A container commonly has none -- the syscall returns ENOENT with no PMU attached --
 * and Android normally sets perf_event_paranoid to 3, which denies it outright. So this is
 * probed, never assumed, and the driver is TOLD which happened. A harness that silently
 * reported nanoseconds under a "cycles" heading would be the same class of error as a table
 * built from an unmeasured target, which is the thing this whole phase exists to prevent.
 */
#if defined(__linux__)
#define BCIR_BENCH_COUNTERS 1
#include <errno.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>


static int cycles_fd = -1;
static const char *counters_state = "not attempted";

static void counters_open(void) {
  struct perf_event_attr attr;
  memset(&attr, 0, sizeof attr);
  attr.type = PERF_TYPE_HARDWARE;
  attr.size = sizeof attr;
  attr.config = PERF_COUNT_HW_CPU_CYCLES;
  attr.disabled = 1;
  /* User-space only. Kernel and hypervisor cycles are not this parser's cost, and counting
   * them would make the figure depend on what else the machine was doing. */
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;
  cycles_fd = (int)syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
  if (cycles_fd < 0) {
    counters_state = strerror(errno);
    return;
  }
  ioctl(cycles_fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(cycles_fd, PERF_EVENT_IOC_ENABLE, 0);
  counters_state = "cycles";
}

static uint64_t now_cycles(void) {
  uint64_t value = 0;
  if (cycles_fd < 0) return 0;
  if (read(cycles_fd, &value, sizeof value) != (ssize_t)sizeof value) return 0;
  return value;
}
#else
static int cycles_fd = -1;
static const char *counters_state = "not linux";
static void counters_open(void) {}
static uint64_t now_cycles(void) { return 0; }
#endif

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
  char op[16];
  unsigned char data[MAX_BYTES];
  size_t len;
  /* An ENCODE case carries a descriptor as well as a value; `encode` selects the arm. The
   * two live in one struct so the interleaved round-robin below can mix decode and encode
   * cases in a single run -- which matters, because comparing an encode number measured in
   * one process against a decode number measured in another reintroduces exactly the drift
   * the interleaving exists to remove. */
  int encode;
  /* A SCHEMA-DIRECTED decode case (`dircase`). Distinct from `encode` and from the
   * schema-free decode arm because it answers a third question: what does decode cost when
   * the type is already in hand, which is the only way X.696 6.2 lets OER be decoded at all.
   * The schema-free arm measures whether untrusted octets can be walked WITHOUT a type; the
   * two are different costs and are never averaged into one row. */
  int directed;
  /* Which plan-driven decoder answers this case. OER and PER both have one now, and they take
   * different field tables -- X.696's is octet-oriented with a width and a sign, X.691's is
   * bit-oriented with two bounds -- so the case carries both arrays rather than a union that
   * would need a tag to read safely anyway. */
  int directed_per;
  int per_aligned;
  size_t field_count;
  bcir_oer_field fields[MAX_PLAN_MEMBERS];
  bcir_per_field per_fields[MAX_PLAN_MEMBERS];
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
  if (c->directed && c->directed_per) {
    /* X.691's plan-driven decoder. `aligned` is on the CALL rather than on a field because
     * `bcir_per_align` is the only difference between the two variants at a field boundary --
     * and it is at every boundary, so one table serves both. */
    bcir_per_value out[MAX_PLAN_MEMBERS];
    size_t end_bit = 0;
    sink += (uint64_t)bcir_per_decode_sequence(c->data, c->len, c->per_fields,
                                               c->field_count, c->per_aligned, 0,
                                               out, &end_bit);
    sink += end_bit;
    return sink;
  }
  if (c->directed) {
    /* X.696's. The two decoders are timed through the same `dircase` verb because they answer
     * the same question -- what does decode cost with the type in hand -- and a table that
     * mixed them with the schema-free arm would be averaging two different questions. */
    bcir_oer_value out[MAX_PLAN_MEMBERS];
    bcir_oer_diag diag;
    size_t end = 0;
    int canonical = 0;
    sink += (uint64_t)bcir_oer_decode_sequence(c->data, c->len, 0, c->fields,
                                               c->field_count, out, &end, &canonical, &diag);
    sink += end + (uint64_t)canonical;
    return sink;
  }
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
  static uint64_t rounds_cycles[MAX_CASES][MAX_ROUNDS];
  static uint64_t batch[4096];
  static uint64_t cycle_batch[4096];
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
      if (sscanf(line, "%31s %63s %15s %" MAX_LINE_SCAN "s", op, c->label, c->op, hex) != 4) {
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

    if (strcmp(op, "dircase") == 0) {
      /* dircase <label> <op> <fields-hex> <octets-hex>
       *
       * The field array is built in Python from the same encode plan, so the mapping from an
       * ASN.1 type to X.696's field kinds lives beside the plan semantics rather than being
       * re-derived here from a second parse of the descriptor. */
      bench_case *c;
      long flen, vlen;
      unsigned char fbuf[MAX_PLAN_MEMBERS * 8];
      /* Its own buffer: `plan_hex` belongs to the encase arm, and a field array is not a
       * plan. Sized at MAX_LINE like every other token buffer here, NOT at the record form's
       * 16 hex digits per field: the `per-d` arms below read a TEXT plan through this same
       * buffer, and X.691's field carries two signed 64-bit bounds, so one field can spell
       * ~45 characters rather than 16. At the record-form size a plan of about a hundred
       * members overran it -- a write past the end of a static buffer, from a plan a caller
       * could reasonably write. A token scanned out of `line` cannot exceed `line`, so
       * matching MAX_LINE is what makes the width below sufficient rather than merely large. */
      static char fields_hex[MAX_LINE];
      size_t i;
      if (n_cases >= MAX_CASES) { printf("unsupported dircase too-many\n"); return 2; }
      c = &cases[n_cases];
      if (sscanf(line, "%31s %63s %15s %" MAX_LINE_SCAN "s %" MAX_LINE_SCAN "s",
                 op, c->label, c->op, fields_hex, hex) != 5) {
        printf("unsupported dircase malformed\n");
        return 2;
      }
      vlen = unhex(hex, c->data, sizeof(c->data));
      if (vlen < 0) { printf("unsupported dircase bad-hex\n"); return 2; }
      c->directed = 1;
      c->len = (size_t)vlen;

      if (strncmp(c->op, "per-d", 5) == 0) {
        /* X.691's field carries two signed 64-bit bounds, so the plan arrives as text --
         * `kind:bounds:lb:ub:fixed:optional`, comma separated, the same spelling
         * test_per_plan.c reads. Sixteen bytes of endian-sensitive record would be the
         * alternative, for a benchmark argument. */
        char *tok = fields_hex;
        c->directed_per = 1;
        c->per_aligned = (strcmp(c->op, "per-d-aligned") == 0);
        c->field_count = 0;
        while (tok != 0 && *tok != '\0' && c->field_count < MAX_PLAN_MEMBERS) {
          int kind, bounds, optional;
          long long lb, ub;
          unsigned fixed;
          char *comma = strchr(tok, ',');
          if (comma != 0) *comma = '\0';
          if (sscanf(tok, "%d:%d:%lld:%lld:%u:%d",
                     &kind, &bounds, &lb, &ub, &fixed, &optional) != 6) {
            printf("unsupported dircase bad-plan\n");
            return 2;
          }
          c->per_fields[c->field_count].kind = (bcir_per_kind)kind;
          c->per_fields[c->field_count].bounds = (bcir_per_bounds)bounds;
          c->per_fields[c->field_count].lb = (int64_t)lb;
          c->per_fields[c->field_count].ub = (int64_t)ub;
          c->per_fields[c->field_count].fixed_len = (uint32_t)fixed;
          c->per_fields[c->field_count].optional = (uint8_t)optional;
          c->field_count++;
          tok = (comma != 0) ? comma + 1 : 0;
        }
        if (c->field_count == 0) { printf("unsupported dircase bad-plan\n"); return 2; }
        n_cases++;
        continue;
      }

      flen = unhex(fields_hex, fbuf, sizeof(fbuf));
      if (flen < 0 || (flen % 8) != 0 || (size_t)(flen / 8) > MAX_PLAN_MEMBERS) {
        printf("unsupported dircase bad-hex\n");
        return 2;
      }
      c->field_count = (size_t)(flen / 8);
      for (i = 0; i < c->field_count; i++) {
        const unsigned char *r = fbuf + i * 8;
        c->fields[i].kind = (bcir_oer_kind)r[0];
        c->fields[i].width = r[1];
        c->fields[i].is_signed = r[2];
        c->fields[i].optional = r[3];
        c->fields[i].fixed_len = (uint32_t)r[4] | ((uint32_t)r[5] << 8) |
                                 ((uint32_t)r[6] << 16) | ((uint32_t)r[7] << 24);
      }
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
      if (sscanf(line, "%31s %63s %15s %" MAX_LINE_SCAN "s %" MAX_LINE_SCAN "s",
                 op, c->label, c->op, plan_hex, hex) != 5) {
        printf("unsupported encase malformed\n");
        return 2;
      }
      if (strcmp(c->op, "der") == 0) c->rules = BCIR_EMIT_DER;
      else if (strcmp(c->op, "ber") == 0) c->rules = BCIR_EMIT_BER;
      else if (strcmp(c->op, "jer") == 0) c->rules = BCIR_EMIT_JER;
      else if (strcmp(c->op, "coer") == 0) c->rules = BCIR_EMIT_COER;
      else if (strcmp(c->op, "cper-a") == 0) c->rules = BCIR_EMIT_CPER_ALIGNED;
      else if (strcmp(c->op, "cper-u") == 0) c->rules = BCIR_EMIT_CPER_UNALIGNED;
      else if (strcmp(c->op, "bper-a") == 0) c->rules = BCIR_EMIT_BPER_ALIGNED;
      else if (strcmp(c->op, "bper-u") == 0) c->rules = BCIR_EMIT_BPER_UNALIGNED;
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

  /* Probed once, before anything is timed, so every round shares one counter -- and before
   * the empty-corpus exit, so a caller can ask what this host offers without supplying one. */
  counters_open();
  printf("counters %s\n", counters_state);

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
      /* GROUPS, not single iterations, and this is the difference between a number and a
       * tick count.
       *
       * Timing each iteration on its own means every sample is rounded to whatever the
       * clock's period is -- and a median of quantized samples is still quantized, so the
       * batch median inherits the rounding rather than averaging it out. The first aarch64
       * calibration taken with this harness showed it plainly: every distinct figure in the
       * record, 104/156/208/260/417, was an exact multiple of 52.083 ns, the period of the
       * 19.2 MHz ARM architectural timer. Those were 2, 3, 4, 5 and 8 TICKS. A cost this
       * short simply cannot be resolved one iteration at a time on that clock.
       *
       * Timing a GROUP of `group` iterations in one span and dividing spreads the clock's
       * error over the whole group, so the resolution improves by a factor of `group`. The
       * median is then taken over the groups, which keeps the property the per-iteration
       * version was written for: one preempted span moves one group, not the round.
       *
       * `group` is chosen so there are enough groups for that median to mean something. It
       * is a pure resolution win with no change to what is being measured -- the same work,
       * the same order, the same interleaving. */
      long group = iterations / 8;
      long groups;
      if (group < 1) group = 1;
      groups = iterations / group;
      for (long g = 0; g < groups; g++) {
        uint64_t c0 = now_cycles();
        uint64_t t0 = now_ns();
        for (long it = 0; it < group; it++) sink += do_work(&cases[i]);
        batch[g] = (now_ns() - t0) / (uint64_t)group;
        cycle_batch[g] = (now_cycles() - c0) / (uint64_t)group;
      }
      qsort(batch, (size_t)groups, sizeof(batch[0]), cmp_u64);
      qsort(cycle_batch, (size_t)groups, sizeof(cycle_batch[0]), cmp_u64);
      rounds_ns[i][r] = batch[groups / 2];
      rounds_cycles[i][r] = cycle_batch[groups / 2];
    }
  }

  for (size_t i = 0; i < n_cases; i++)
    for (long r = 0; r < rounds; r++)
      printf("sample %s %s %ld %llu %llu\n", cases[i].label, cases[i].op, r,
             (unsigned long long)rounds_ns[i][r],
             (unsigned long long)rounds_cycles[i][r]);
  printf("done %lu\n", (unsigned long)n_cases);
  /* Consume the sink so no call above is dead. */
  if (sink == 0xFFFFFFFFFFFFFFFFull) printf("sink %llu\n", (unsigned long long)sink);
  return 0;
}
