/*===- test_emit.c - line-protocol driver for the plan-driven encoder ------===
 *
 * Hosted, and a TEST driver only: bcir_emit.c itself is freestanding. The protocol is the
 * same shape the other twins use, so the Python differential can drive it without a second
 * marshalling convention to get wrong.
 *
 * Input, one command per line:
 *
 *   plan <hex>                    the serialized descriptor
 *   emit <rules> <hex>            rules is der | ber | jer | coer; hex is the value stream
 *   emitcap <rules> <cap> <hex>   the same with a deliberately small output buffer
 *   scratchcap <n>                cap the size scratch, to exercise SCRATCH_SHORT
 *   constraint <node>             read back what the parser stored for one node
 *   enum <node>                   read back a node's extension marker and enumeration
 *
 * `constraint` exists because plan version 3 records more than any emitter here consumes
 * yet: the X.691 extension-root bounds and the permitted alphabet are PER's, and PER's
 * writer is a later step. A field nothing reads is a field nothing checks, so the Python
 * differential reads them back and compares against the compiler that wrote them.
 *
 * Output:
 *
 *   ok <hex>                      the octets emitted
 *   ok <fields...>                for `constraint`
 *   err <status> <offset> <needed>
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <string.h>

#include "bcir_emit.h"

#define MAX_TEXT (1 << 18)
#define MAX_LINE (MAX_TEXT * 2 + 64)
#define MAX_NODES 512
#define MAX_MEMBERS 512
#define MAX_CONSTRAINTS 64
#define MAX_ENUMS 128
#define MAX_OUT (1 << 20)
#define MAX_SCRATCH 8192

static int hex_nibble(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* `-` spells the EMPTY stream. A SEQUENCE whose only member is NULL flattens to zero
 * octets, and a driver that could not express that would leave the emitters' NULL paths
 * untested -- which is where three encoders were already found to disagree. */
static long unhex(const char *text, unsigned char *out, size_t cap) {
  size_t n = 0;
  if (text[0] == '-' && (text[1] == '\0' || text[1] == '\n' || text[1] == '\r')) return 0;
  while (*text != '\0' && *text != '\n' && *text != '\r') {
    int hi = hex_nibble((unsigned char)text[0]);
    int lo = text[1] ? hex_nibble((unsigned char)text[1]) : -1;
    if (hi < 0 || lo < 0 || n >= cap) return -1;
    out[n++] = (unsigned char)((hi << 4) | lo);
    text += 2;
  }
  return (long)n;
}

static void print_hex(const unsigned char *data, size_t len) {
  size_t i;
  for (i = 0; i < len; i++) printf("%02x", data[i]);
}

int main(void) {
  static char line[MAX_LINE];
  static char hex[MAX_LINE];
  static unsigned char text[MAX_TEXT];
  static unsigned char stream[MAX_TEXT];
  static unsigned char out[MAX_OUT];
  static uint32_t scratch[MAX_SCRATCH];
  static bcir_emit_node nodes[MAX_NODES];
  static bcir_emit_member members[MAX_MEMBERS];
  static bcir_emit_constraint constraints[MAX_CONSTRAINTS];
  static bcir_emit_enum_item enums[MAX_ENUMS];
  bcir_emit_plan plan;
  bcir_emit_tables tables;
  bcir_emit_diag diag;
  int have_plan = 0;
  size_t scratch_cap = MAX_SCRATCH;

  memset(&plan, 0, sizeof(plan));
  while (fgets(line, (int)sizeof(line), stdin) != NULL) {
    char op[32];
    if (sscanf(line, "%31s", op) != 1) continue;

    if (strcmp(op, "plan") == 0) {
      long len;
      bcir_emit_status status;
      if (sscanf(line, "%31s %s", op, hex) != 2) { printf("err 1 0 0\n"); continue; }
      len = unhex(hex, text, sizeof(text));
      if (len < 0) { printf("err 1 0 0\n"); continue; }
      tables.nodes = nodes;             tables.node_cap = MAX_NODES;
      tables.members = members;         tables.member_cap = MAX_MEMBERS;
      tables.constraints = constraints; tables.constraint_cap = MAX_CONSTRAINTS;
      tables.enums = enums;             tables.enum_cap = MAX_ENUMS;
      status = bcir_emit_parse_plan((const char *)text, (size_t)len, &tables, &plan, &diag);
      if (status != BCIR_EMIT_OK) {
        printf("err %d %lu %lu\n", (int)status, (unsigned long)diag.offset,
               (unsigned long)diag.needed);
        have_plan = 0;
      } else {
        have_plan = 1;
        printf("ok\n");
      }
      continue;
    }

    if (strcmp(op, "constraint") == 0) {
      unsigned long index = 0;
      const bcir_emit_node *node;
      const bcir_emit_constraint *k;
      const bcir_emit_bound *bounds[8];
      int i;
      if (sscanf(line, "%31s %lu", op, &index) != 2 || !have_plan ||
          index >= plan.node_count) {
        printf("err 1 0 0\n");
        continue;
      }
      node = &plan.nodes[index];
      if (node->constraint < 0) { printf("ok none\n"); continue; }
      k = &plan.constraints[node->constraint];
      bounds[0] = &k->value_low;      bounds[1] = &k->value_high;
      bounds[2] = &k->size_low;       bounds[3] = &k->size_high;
      bounds[4] = &k->root_value_low; bounds[5] = &k->root_value_high;
      bounds[6] = &k->root_size_low;  bounds[7] = &k->root_size_high;
      printf("ok");
      for (i = 0; i < 8; i++) {
        if (!bounds[i]->present) printf(" -");
        else printf(" %s%llu", bounds[i]->negative ? "-" : "",
                    (unsigned long long)bounds[i]->magnitude);
      }
      printf(" %u %u ", (unsigned)k->value_extensible, (unsigned)k->size_extensible);
      if (k->alphabet_len == 0) printf("-");
      else print_hex((const unsigned char *)k->alphabet, k->alphabet_len);
      printf("\n");
      continue;
    }

    if (strcmp(op, "enum") == 0) {
      unsigned long index = 0;
      const bcir_emit_node *node;
      uint32_t k;
      if (sscanf(line, "%31s %lu", op, &index) != 2 || !have_plan ||
          index >= plan.node_count) {
        printf("err 1 0 0\n");
        continue;
      }
      node = &plan.nodes[index];
      printf("ok %u", (unsigned)node->extensible);
      for (k = 0; k < node->enum_count; k++) {
        const bcir_emit_enum_item *item = &plan.enums[node->first_enum + k];
        printf(" %.*s:%lld", (int)item->name_len, item->name, (long long)item->number);
      }
      printf("\n");
      continue;
    }

    if (strcmp(op, "scratchcap") == 0) {
      unsigned long value = 0;
      if (sscanf(line, "%31s %lu", op, &value) != 2) { printf("err 1 0 0\n"); continue; }
      scratch_cap = value > MAX_SCRATCH ? MAX_SCRATCH : (size_t)value;
      printf("ok\n");
      continue;
    }

    if (strcmp(op, "emit") == 0 || strcmp(op, "emitcap") == 0) {
      char rules_name[16];
      unsigned long cap = MAX_OUT;
      long len;
      size_t written = 0;
      bcir_emit_rules rules;
      bcir_emit_status status;
      int parsed;

      if (strcmp(op, "emit") == 0)
        parsed = sscanf(line, "%31s %15s %s", op, rules_name, hex) == 3;
      else
        parsed = sscanf(line, "%31s %15s %lu %s", op, rules_name, &cap, hex) == 4;
      if (!parsed || !have_plan) { printf("err 1 0 0\n"); continue; }

      if (strcmp(rules_name, "der") == 0) rules = BCIR_EMIT_DER;
      else if (strcmp(rules_name, "ber") == 0) rules = BCIR_EMIT_BER;
      else if (strcmp(rules_name, "jer") == 0) rules = BCIR_EMIT_JER;
      else if (strcmp(rules_name, "coer") == 0) rules = BCIR_EMIT_COER;
      else { printf("err 9 0 0\n"); continue; }

      len = unhex(hex, stream, sizeof(stream));
      if (len < 0) { printf("err 1 0 0\n"); continue; }
      if (cap > MAX_OUT) cap = MAX_OUT;

      status = bcir_emit(&plan, rules, stream, (size_t)len, out, (size_t)cap, &written,
                         scratch, scratch_cap, 32, &diag);
      if (status != BCIR_EMIT_OK) {
        printf("err %d %lu %lu\n", (int)status, (unsigned long)diag.offset,
               (unsigned long)diag.needed);
        continue;
      }
      printf("ok ");
      print_hex(out, written);
      printf("\n");
      continue;
    }
  }
  return 0;
}
