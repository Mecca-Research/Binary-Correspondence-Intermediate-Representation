/*===- test_telemetry_frame.c - Python<->C parity harness for the frame codec ===
 *
 * Reads a Python-encoded telemetry frame (argv[1]), decodes + validates it (the C rail
 * sees what the Python rail wrote), prints the records, and re-encodes the SAME frame
 * through the C writer into argv[2] (if given) so the Python side can assert the bytes
 * are byte-identical to encode_frame (the dual-rail headline). Uses libc (a host
 * harness), like test_encode.c; the codec itself (bcir_telemetry_frame.c) is freestanding.
 *
 *   clang -std=c23 test_telemetry_frame.c bcir_telemetry_frame.c bcir_runtime.c -I . -o t
 *   ./t frame.bin reenc.bin
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>

#include "bcir_telemetry_frame.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <in-frame> [out-reencoded]\n", argv[0]);
    return 2;
  }
  FILE *fp = fopen(argv[1], "rb");
  if (!fp) { perror("fopen in"); return 2; }
  static uint8_t in[1 << 20];
  size_t in_len = fread(in, 1, sizeof in, fp);
  fclose(fp);

  bcir_tf_header hdr;
  size_t frame_len = 0;
  bcir_tf_status st = bcir_tf_decode_frame(in, in_len, &hdr, &frame_len);
  if (st != BCIR_TF_OK) { printf("ERR %d\n", (int)st); return 1; }

  /* Print the decoded header + every record so the Python rail can diff the values. */
  printf("seq=%u ts=%llu n=%u\n", hdr.seq,
         (unsigned long long)hdr.timestamp, (unsigned)hdr.n_records);
  static bcir_tf_record recs[1 << 14];
  for (uint16_t i = 0; i < hdr.n_records; i++) {
    bcir_tf_status rs = bcir_tf_get_record(in, frame_len, i, &recs[i]);
    if (rs != BCIR_TF_OK) { printf("ERR-REC %d\n", (int)rs); return 1; }
    printf("rec %u: %lld %lld %lld %lld %lld %lld %lld\n", (unsigned)i,
           (long long)recs[i].claim_id, (long long)recs[i].cycles,
           (long long)recs[i].bytes, (long long)recs[i].misses,
           (long long)recs[i].thermal, (long long)recs[i].voltage,
           (long long)recs[i].utilization);
  }

  /* Re-encode the SAME frame (same seq/ts/records) through the C writer; the Python
   * test cmp's it against the original encode_frame bytes (byte-identical dual-rail). */
  if (argc >= 3) {
    static uint8_t out[1 << 20];
    size_t out_len = bcir_tf_encode_frame(recs, hdr.n_records, hdr.seq, hdr.timestamp,
                                          out, sizeof out);
    if (out_len == 0) { printf("ERR-ENC nospace\n"); return 1; }
    FILE *op = fopen(argv[2], "wb");
    if (!op) { perror("fopen out"); return 2; }
    if (fwrite(out, 1, out_len, op) != out_len) { fclose(op); return 2; }
    fclose(op);
  }
  printf("OK in=%zu frame=%zu\n", in_len, frame_len);
  return 0;
}
