/*===- test_per_plan.c - a line protocol over the plan-driven PER decoder --------------------===
 *
 * Same shape as test_oer.c: stdin lines, one construct each, so the differential in
 * bcir/tests/ can drive the C twin without a second encoder living here.
 *
 *   sequence <hex> <aligned> <extensible> <plan>
 *
 * `<plan>` is a comma-separated field list, each `kind:bounds:lb:ub:fixed:optional`, which is
 * every property bcir_per_field carries. A test can therefore build any schema the decoder
 * claims to support, and a schema it does NOT claim produces a refusal rather than a misread.
 *
 * Output is "OK <endbit> <n> <v0> <v1> ..." where each value is `present:integer:offset:length`,
 * or "ERR <status>".
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <string.h>

#include "bcir_per_plan.h"

#define MAX_FIELDS 32

static int unhex(const char *s, uint8_t *out, size_t cap, size_t *len) {
  size_t n = strlen(s), i;
  if (n % 2u != 0u || n / 2u > cap) return 0;
  for (i = 0; i < n; i += 2) {
    unsigned byte;
    if (sscanf(s + i, "%2x", &byte) != 1) return 0;
    out[i / 2u] = (uint8_t)byte;
  }
  *len = n / 2u;
  return 1;
}

/* `kind:bounds:lb:ub:fixed:optional`, comma separated. Parsed strictly: a field that does not
 * carry all six is a malformed test rather than a defaulted one, because a default here would
 * silently change what the decoder was asked to do. */
static size_t parse_plan(char *text, bcir_per_field *fields) {
  size_t count = 0;
  char *tok = text;
  /* Split on ',' by hand rather than with strtok_r, which is POSIX and needs a feature macro
   * this file has no other reason to define. The harness stays strictly conforming C. */
  while (tok != 0 && *tok != '\0' && count < MAX_FIELDS) {
    int kind, bounds, optional;
    long long lb, ub;
    unsigned fixed;
    char *comma = strchr(tok, ',');
    if (comma != 0) *comma = '\0';
    if (sscanf(tok, "%d:%d:%lld:%lld:%u:%d", &kind, &bounds, &lb, &ub, &fixed, &optional) != 6)
      return 0;
    fields[count].kind = (bcir_per_kind)kind;
    fields[count].bounds = (bcir_per_bounds)bounds;
    fields[count].lb = (int64_t)lb;
    fields[count].ub = (int64_t)ub;
    fields[count].fixed_len = (uint32_t)fixed;
    fields[count].optional = (uint8_t)optional;
    ++count;
    tok = (comma != 0) ? comma + 1 : 0;
  }
  return count;
}

int main(void) {
  char line[8192];
  while (fgets(line, sizeof line, stdin) != 0) {
    char verb[32], hex[4096], plan[2048];
    int aligned = 0, extensible = 0;
    if (sscanf(line, "%31s", verb) != 1) continue;
    if (strcmp(verb, "sequence") == 0) {
      bcir_per_field fields[MAX_FIELDS];
      bcir_per_value values[MAX_FIELDS];
      uint8_t buf[2048];
      size_t len = 0, count, endbit = 0, i;
      bcir_per_status st;
      if (sscanf(line, "%31s %4095s %d %d %2047s", verb, hex, &aligned, &extensible, plan) != 5) {
        printf("ERR 4\n");
        continue;
      }
      if (!unhex(hex, buf, sizeof buf, &len)) { printf("ERR 4\n"); continue; }
      count = parse_plan(plan, fields);
      if (count == 0) { printf("ERR 4\n"); continue; }
      st = bcir_per_decode_sequence(buf, len, fields, count, aligned, extensible,
                                    values, &endbit);
      if (st != BCIR_PER_OK) { printf("ERR %d\n", (int)st); continue; }
      printf("OK %zu %zu", endbit, count);
      for (i = 0; i < count; ++i)
        printf(" %d:%lld:%zu:%zu", values[i].present, (long long)values[i].integer,
               values[i].offset, values[i].length);
      printf("\n");
    }
  }
  return 0;
}
