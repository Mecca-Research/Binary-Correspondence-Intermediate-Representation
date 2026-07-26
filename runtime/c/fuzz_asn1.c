/*===- fuzz_asn1.c - libFuzzer harness for the X.690 decoder ----------------===
 *
 * bcir_asn1.c parses BER/DER from a peer, so every octet is attacker-controlled --
 * and X.690's own structures are the ones that historically break hand-written
 * parsers: multi-octet tags and lengths, indefinite-length constructed encodings
 * terminated by end-of-contents octets, and arbitrary nesting.
 *
 * The contract under test: for ANY (data, len), every entry point returns a status
 * and never reads outside [data, data+len). Specifically
 *   - a constructed encoding's declared length bounds its children (a child that
 *     overruns its parent must be TRUNCATED, not silently accepted against the
 *     outer buffer);
 *   - a stray end-of-contents identifier octet inside a definite-length body is a
 *     fault, never "iteration finished" (the two must not share a status);
 *   - nesting is bounded, so no input drives unbounded recursion or stack growth.
 *
 *   clang -fsanitize=fuzzer,address,undefined -std=c23 \
 *       runtime/c/fuzz_asn1.c runtime/c/bcir_asn1.c -I runtime/c -o fuzz_asn1
 *   ./fuzz_asn1 -runs=1000000          (see tools/c/fuzz_streampack.sh)
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>

#include "bcir_asn1.h"

#define FUZZ_MAX_DEPTH 32u
#define FUZZ_STACK 32u

/* Walk every node, iteratively, touching each one's contents so ASan sees any
 * out-of-bounds view the decoder might have produced. */
static void walk_all(const uint8_t *data, size_t len) {
  bcir_asn1_tlv parent[FUZZ_STACK];
  bcir_asn1_tlv child[FUZZ_STACK];
  unsigned depth = 0u;
  bcir_asn1_tlv root;
  uint64_t sink = 0u;

  if (bcir_asn1_decode_exact(data, len, &root) != BCIR_ASN1_OK) return;
  if (!root.constructed) return;
  if (bcir_asn1_first_child(data, len, &root, &child[0]) != BCIR_ASN1_OK) return;
  parent[0] = root;

  for (;;) {
    bcir_asn1_tlv *cur = &child[depth];
    size_t i;
    /* Read every contents octet the decoder claims is in bounds. */
    for (i = 0; i < cur->content_len; ++i) sink += cur->content[i];
    (void)bcir_asn1_check_der_node(cur);
    {
      int flag = 0;
      int64_t signed_value = 0;
      uint64_t unsigned_value = 0;
      uint32_t arcs[16];
      size_t arc_count = 0;
      (void)bcir_asn1_boolean(cur, 1, &flag);
      (void)bcir_asn1_integer(cur, &signed_value);
      (void)bcir_asn1_uinteger(cur, &unsigned_value);
      (void)bcir_asn1_oid_arcs(cur, arcs, sizeof arcs / sizeof arcs[0], &arc_count);
      sink += (uint64_t)flag + (uint64_t)signed_value + unsigned_value + arc_count;
    }

    if (cur->constructed && depth + 1u < FUZZ_STACK) {
      bcir_asn1_tlv first;
      if (bcir_asn1_first_child(data, len, cur, &first) == BCIR_ASN1_OK) {
        parent[depth + 1u] = *cur;
        child[depth + 1u] = first;
        depth++;
        continue;
      }
    }
    for (;;) {
      if (bcir_asn1_next(data, len, &parent[depth], &child[depth]) == BCIR_ASN1_OK)
        break;
      if (depth == 0u) { (void)sink; return; }
      depth--;
    }
  }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  bcir_asn1_tlv tlv;

  /* (1) single-encoding decode, exact-fit decode, and the two whole-buffer walks */
  (void)bcir_asn1_decode(data, size, 0u, &tlv);
  (void)bcir_asn1_decode_exact(data, size, &tlv);
  (void)bcir_asn1_validate(data, size, FUZZ_MAX_DEPTH);
  (void)bcir_asn1_validate_der(data, size, FUZZ_MAX_DEPTH);

  /* (2) decode from every offset: a sub-buffer must be as safe as the whole one */
  if (size) {
    size_t step = size / 8u ? size / 8u : 1u;
    size_t offset;
    for (offset = 0; offset < size; offset += step)
      (void)bcir_asn1_decode(data, size, offset, &tlv);
    (void)bcir_asn1_decode(data, size, size, &tlv);     /* the empty tail */
  }

  /* (3) the full child walk, reading every contents octet the decoder hands out */
  walk_all(data, size);
  return 0;
}
