/*===- bcir_telemetry_frame.c - freestanding BCIR telemetry-frame codec ----===
 *
 * The C twin of bcir/telemetry_frame.py. No libc: only <stddef.h>/<stdint.h> (via the
 * header). Little-endian wire encoding/decoding done byte by byte (host-endian-
 * independent), matching the Python rail exactly. The CRC-32 is REUSED from
 * bcir_runtime.c (bcir_crc32) -- never reimplemented -- so both rails' CRCs agree.
 *===----------------------------------------------------------------------===*/
#include "bcir_telemetry_frame.h"
#include "bcir_bytes.h"

/* Decode a two's-complement wire pattern without relying on the implementation-defined
 * conversion from an out-of-range uint64_t to int64_t in C11/C17. The subtraction is
 * always in range before the signed cast, including the INT64_MIN bit pattern. */
static int64_t rd_i64(const uint8_t *p) {
  uint64_t bits = bcir_ld64le(p);
  if (bits <= (uint64_t)INT64_MAX)
    return (int64_t)bits;
  return -INT64_C(1) - (int64_t)(UINT64_MAX - bits);
}

/* Write one 56-byte record (the <7q> ring layout) at `p`, explicit little-endian. The
 * int64 fields are reinterpreted as their two's-complement u64 bit pattern (the same
 * bytes struct.pack("<q", ...) emits), so a negative counter round-trips byte-identical
 * to the Python rail. */
static void wr_record(uint8_t *p, const bcir_tf_record *r) {
  bcir_st64le(p +  0, (uint64_t)r->claim_id);
  bcir_st64le(p +  8, (uint64_t)r->cycles);
  bcir_st64le(p + 16, (uint64_t)r->bytes);
  bcir_st64le(p + 24, (uint64_t)r->misses);
  bcir_st64le(p + 32, (uint64_t)r->thermal);
  bcir_st64le(p + 40, (uint64_t)r->voltage);
  bcir_st64le(p + 48, (uint64_t)r->utilization);
}

size_t bcir_tf_encode_frame(const bcir_tf_record *BCIR_TF_RESTRICT recs, uint16_t n,
                            uint32_t seq, uint64_t ts, uint8_t *BCIR_TF_RESTRICT out,
                            size_t out_len) {
  size_t need = 0;
  if (!bcir_tf_frame_size_checked(n, &need) || !out || (n && !recs) || out_len < need)
    return 0; /* unrepresentable / invalid / NOSPACE */

  /* header (22 bytes): magic[4], version:u16, flags:u16, seq:u32, timestamp:u64, n:u16 */
  out[0] = 'B'; out[1] = 'T'; out[2] = 'L'; out[3] = 'M';
  bcir_st16le(out + 4, (uint16_t)BCIR_TELEMETRY_FRAME_VERSION);
  bcir_st16le(out + 6, 0u);                             /* flags reserved */
  bcir_st32le(out + 8, seq);
  bcir_st64le(out + 12, ts);
  bcir_st16le(out + 20, n);

  /* body: n records in the shared 56-byte <7q> layout, serialized explicitly LE */
  size_t off = BCIR_TELEMETRY_FRAME_HEADER_SIZE;
  for (uint16_t i = 0; i < n; i++) {
    wr_record(out + off, &recs[i]);
    off += BCIR_TF_RECORD_SIZE;
  }

  /* trailer: CRC-32 over every preceding byte (REUSED bcir_crc32 == zlib.crc32) */
  uint32_t crc = bcir_crc32(out, off);
  bcir_st32le(out + off, crc);
  return need;
}

bcir_tf_status bcir_tf_decode_frame(const uint8_t *BCIR_TF_RESTRICT buf, size_t len,
                                    bcir_tf_header *hdr, size_t *frame_len) {
  if (hdr) {
    hdr->version = 0; hdr->flags = 0; hdr->seq = 0;
    hdr->timestamp = 0; hdr->n_records = 0;
  }
  if (frame_len) *frame_len = 0;
  if (!buf) return BCIR_TF_ERR_TRUNCATED;
  /* A frame is at least header + CRC (an empty, well-formed frame). */
  if (len < (size_t)BCIR_TELEMETRY_FRAME_HEADER_SIZE + BCIR_TELEMETRY_FRAME_CRC_SIZE)
    return BCIR_TF_ERR_TRUNCATED;
  if (buf[0] != 'B' || buf[1] != 'T' || buf[2] != 'L' || buf[3] != 'M')
    return BCIR_TF_ERR_MAGIC;
  uint16_t version = bcir_ld16le(buf + 4);
  if (version < 1u || version > (uint16_t)BCIR_TELEMETRY_FRAME_VERSION)
    return BCIR_TF_ERR_VERSION;
  if (bcir_ld16le(buf + 6) != 0u) return BCIR_TF_ERR_FLAGS;
  uint16_t n = bcir_ld16le(buf + 20);
  size_t need = 0;
  if (!bcir_tf_frame_size_checked(n, &need) || need > len)
    return BCIR_TF_ERR_TRUNCATED;  /* unrepresentable or claims more records than fit */

  size_t body_len = (size_t)BCIR_TELEMETRY_FRAME_HEADER_SIZE
                  + (size_t)n * (size_t)BCIR_TF_RECORD_SIZE;
  uint32_t stored = bcir_ld32le(buf + body_len);
  if (bcir_crc32(buf, body_len) != stored) return BCIR_TF_ERR_CRC;

  if (hdr) {
    hdr->version = version;
    hdr->flags = bcir_ld16le(buf + 6);
    hdr->seq = bcir_ld32le(buf + 8);
    hdr->timestamp = bcir_ld64le(buf + 12);
    hdr->n_records = n;
  }
  if (frame_len) *frame_len = need;
  return BCIR_TF_OK;
}

bcir_tf_status bcir_tf_decode_exact(const uint8_t *BCIR_TF_RESTRICT buf, size_t len,
                                    bcir_tf_header *hdr) {
  size_t used = 0;
  bcir_tf_header decoded;
  if (hdr) {
    hdr->version = 0; hdr->flags = 0; hdr->seq = 0;
    hdr->timestamp = 0; hdr->n_records = 0;
  }
  bcir_tf_status st = bcir_tf_decode_frame(buf, len, &decoded, &used);
  if (st != BCIR_TF_OK) return st;
  if (used != len) return BCIR_TF_ERR_TRAILING;
  if (hdr) *hdr = decoded;
  return BCIR_TF_OK;
}

bcir_tf_status bcir_tf_get_record(const uint8_t *BCIR_TF_RESTRICT buf, size_t len,
                                  uint16_t i, bcir_tf_record *out) {
  bcir_tf_header header;
  size_t frame_size = 0;
  bcir_tf_status status;
  if (out) {
    out->claim_id = 0; out->cycles = 0; out->bytes = 0; out->misses = 0;
    out->thermal = 0; out->voltage = 0; out->utilization = 0;
  }
  /* Validate the frame on every standalone accessor call. Requiring callers to
   * remember a prior validation allowed CRC-corrupt bytes (or a buffer modified
   * after validation) to be returned as trusted records. */
  status = bcir_tf_decode_frame(buf,len,&header,&frame_size);
  if (status != BCIR_TF_OK) return status;
  if (i >= header.n_records) return BCIR_TF_ERR_RANGE;
  size_t off = (size_t)BCIR_TELEMETRY_FRAME_HEADER_SIZE
             + (size_t)i * (size_t)BCIR_TF_RECORD_SIZE;
  if (off > frame_size || BCIR_TF_RECORD_SIZE > frame_size - off)
    return BCIR_TF_ERR_TRUNCATED;
  if (out) {
    out->claim_id    = rd_i64(buf + off +  0);
    out->cycles      = rd_i64(buf + off +  8);
    out->bytes       = rd_i64(buf + off + 16);
    out->misses      = rd_i64(buf + off + 24);
    out->thermal     = rd_i64(buf + off + 32);
    out->voltage     = rd_i64(buf + off + 40);
    out->utilization = rd_i64(buf + off + 48);
  }
  return BCIR_TF_OK;
}

size_t bcir_tf_find_magic(const uint8_t *BCIR_TF_RESTRICT buf, size_t len, size_t start) {
  if (!buf || start >= len || len < 4u) return len;
  /* need 4 bytes for the magic */
  for (size_t i = start; i <= len - 4u; i++) {
    if (buf[i] == 'B' && buf[i + 1] == 'T' && buf[i + 2] == 'L' && buf[i + 3] == 'M')
      return i;
  }
  return len;
}
