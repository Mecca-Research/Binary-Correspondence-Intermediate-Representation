/*===- test_asn1.c - drive the C X.690 decoder over a file, for rail parity ---===
 *
 * Reads a DER/BER encoding from a file and prints one line per node in document
 * order, so the Python rail can compare structure field for field:
 *
 *   <depth> <class> <number> <constructed> <indefinite> <content_len> <total_len>
 *
 * then a verdict line: "validate <status>" and "der <status>". A malformed input
 * prints the failing status instead of a tree, and still exits 0 -- the harness is
 * comparing verdicts, and a decoder that refuses bad bytes is the behaviour under
 * test, not a harness failure.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>

#include "bcir_asn1.h"

#define MAX_INPUT (1u << 20)
#define MAX_DEPTH 32u

static unsigned char g_buf[MAX_INPUT];

/* Print the subtree rooted at `node`, iteratively (the decoder never recurses and
 * neither does its harness). */
static int print_tree(const unsigned char *data, size_t len) {
  bcir_asn1_tlv parent[MAX_DEPTH];
  bcir_asn1_tlv child[MAX_DEPTH];
  unsigned depth = 0u;
  bcir_asn1_tlv root;
  bcir_asn1_status status = bcir_asn1_decode_exact(data, len, &root);

  if (status != BCIR_ASN1_OK) {
    printf("error %s\n", bcir_asn1_status_name(status));
    return 0;
  }
  printf("0 %d %u %d %d %zu %zu\n", (int)root.cls, root.number, root.constructed,
         root.indefinite, root.content_len, root.total_len);
  if (!root.constructed) return 0;

  parent[0] = root;
  status = bcir_asn1_first_child(data, len, &parent[0], &child[0]);
  if (status == BCIR_ASN1_END) return 0;
  if (status != BCIR_ASN1_OK) { printf("error %s\n", bcir_asn1_status_name(status)); return 0; }

  for (;;) {
    bcir_asn1_tlv *cur = &child[depth];
    printf("%u %d %u %d %d %zu %zu\n", depth + 1u, (int)cur->cls, cur->number,
           cur->constructed, cur->indefinite, cur->content_len, cur->total_len);
    if (cur->constructed && depth + 1u < MAX_DEPTH) {
      bcir_asn1_tlv first;
      status = bcir_asn1_first_child(data, len, cur, &first);
      if (status == BCIR_ASN1_OK) {
        parent[depth + 1u] = *cur;
        child[depth + 1u] = first;
        depth++;
        continue;
      }
      if (status != BCIR_ASN1_END) {
        printf("error %s\n", bcir_asn1_status_name(status));
        return 0;
      }
    }
    for (;;) {
      status = bcir_asn1_next(data, len, &parent[depth], &child[depth]);
      if (status == BCIR_ASN1_OK) break;
      if (status != BCIR_ASN1_END) {
        printf("error %s\n", bcir_asn1_status_name(status));
        return 0;
      }
      if (depth == 0u) return 0;
      depth--;
    }
  }
}

int main(int argc, char **argv) {
  FILE *file;
  size_t len;

  if (argc < 2) { fprintf(stderr, "usage: test_asn1 <file>\n"); return 2; }
  file = fopen(argv[1], "rb");
  if (!file) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
  len = fread(g_buf, 1, sizeof g_buf, file);
  if (ferror(file)) { fclose(file); fprintf(stderr, "read error\n"); return 2; }
  fclose(file);

  print_tree(g_buf, len);
  printf("validate %s\n", bcir_asn1_status_name(bcir_asn1_validate(g_buf, len, MAX_DEPTH)));
  printf("der %s\n", bcir_asn1_status_name(bcir_asn1_validate_der(g_buf, len, MAX_DEPTH)));
  return 0;
}
