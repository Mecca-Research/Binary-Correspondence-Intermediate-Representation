/*===- bcir_telemetry_frame.h - BCIR UART telemetry frame ABI v1 (frozen) --===
 *
 * The freestanding C twin of bcir/telemetry_frame.py (the Python reference). A
 * self-delimiting, resync-able, CRC-sealed TELEMETRY FRAME a bare-metal producer drains
 * from the telemetry ring (TelemetryRing) and emits over a byte egress (UART). The
 * prose spec is docs/TELEMETRY_FRAME_ABI.md; a byte-identical differential test
 * (bcir/tests/test_telemetry_frame.py) + the #telemetry-frame gate in
 * tools/c/check_runtime.sh pin this rail to the Python rail.
 *
 * Depends only on <stddef.h> + <stdint.h> (freestanding-clean, C11 + C23), so it links
 * into a driver/embedded host with no libc. The CRC-32 is REUSED from bcir_runtime.c
 * (bcir_crc32, zlib-compatible poly 0xEDB88320) -- declared extern below, never
 * reimplemented -- so the C and Python rails' CRCs agree byte-for-byte.
 *
 * Wire format (little-endian throughout):
 *   frame := magic[4]    ("BTLM" -- the resync anchor)
 *          | version:u16 (== 1; a v1 reader rejects a newer version)
 *          | flags:u16   (reserved, 0)
 *          | seq:u32      (monotonic frame sequence -- drop/reorder detection)
 *          | timestamp:u64 (opaque producer-local tick; 0 if unavailable)
 *          | n_records:u16
 *          | records[n_records]  (each = the 56-byte <7q> DataDNA record, REUSING the
 *                                 ring layout: claim_id, cycles, bytes, misses, thermal,
 *                                 voltage, utilization -- all int64)
 *          | crc32:u32    (zlib-compatible CRC over every preceding frame byte)
 * The header is 22 bytes; the trailer CRC is 4; a frame with n records is 22+56*n+4.
 *
 * RESYNC-ON-MAGIC: a decoder joins mid-stream / recovers from corruption by scanning to
 * the next "BTLM". A bad/truncated/CRC-failing candidate does not poison later frames;
 * false magic inside corrupt bytes can make candidate rejects exceed originating frames.
 *
 * EGRESS-OVER-UART (documented adapter, NOT a hardware dependency): bcir_tf_encode_frame
 * writes the frame bytes to a caller `out` buffer (the testable core). To egress over
 * hardware, an eventual UART adapter can feed those bytes to its byte sink:
 *     for (size_t i = 0; i < n; i++) uart_send(uart, out[i]);
 * The uart_regs.h + cfront_driver_uart.c pair is only a compiler fixture containing a
 * sample uart_send; it is not a resident driver. No UART is touched here.
 *
 * TWO-TRUTH: the frame carries telemetry DATA (graded L2/L3), never a legality verdict.
 * Decoding yields records + a status, never a Diagnostic; the host RT3 ingest gate
 * (bcir.telemetry.sanitize_events) bounds the blast radius. Off the legality path.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_TELEMETRY_FRAME_H
#define BCIR_TELEMETRY_FRAME_H

#include <stddef.h>
#include <stdint.h>

/* C23 niceties, portably guarded so the freestanding header still builds under C11 and
 * as C++ (matches bcir_streampack.h's macro discipline). */
#if defined(__cplusplus)
#  define BCIR_TF_RESTRICT __restrict
#  define BCIR_TF_NODISCARD [[nodiscard]]
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#  define BCIR_TF_RESTRICT restrict
#  define BCIR_TF_NODISCARD [[nodiscard]]
#else
#  define BCIR_TF_RESTRICT restrict
#  define BCIR_TF_NODISCARD
#endif

#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#  define BCIR_TF_STATIC_ASSERT(c, m) static_assert(c, m)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define BCIR_TF_STATIC_ASSERT(c, m) _Static_assert(c, m)
#else
#  define BCIR_TF_STATIC_ASSERT(c, m)
#endif

#define BCIR_TELEMETRY_FRAME_MAGIC      "BTLM"   /* bytes 0..3 of the frame */
#define BCIR_TELEMETRY_FRAME_VERSION    1
#define BCIR_TELEMETRY_FRAME_HEADER_SIZE 22u     /* magic4 + ver2 + flags2 + seq4 + ts8 + n2 */
#define BCIR_TELEMETRY_FRAME_CRC_SIZE   4u
#define BCIR_TF_RECORD_SIZE             56u      /* 7 x int64 (the ring's <7q> stride) */

#ifdef __cplusplus
extern "C" {
#endif

/* The 56-byte telemetry record -- the frozen <7q> DataDNA layout REUSED from the ring
 * (TelemetryRing._FMT / parse_shared_ring). Field order matches the Python pack exactly.
 * Lock its size at compile time (the frozen record ABI). NOTE: the WIRE encoding packs
 * these fields explicitly little-endian byte by byte (bcir_tf_encode_frame) -- it does
 * NOT rely on this struct's layout/padding, exactly as the StreamPack twin does. */
typedef struct bcir_tf_record {
  int64_t claim_id;
  int64_t cycles;
  int64_t bytes;
  int64_t misses;
  int64_t thermal;
  int64_t voltage;
  int64_t utilization;
} bcir_tf_record;

BCIR_TF_STATIC_ASSERT(sizeof(bcir_tf_record) == BCIR_TF_RECORD_SIZE,
                      "BCIR telemetry record must be exactly 56 bytes (frozen <7q> ABI)");

/* Decoded frame header (continuity metadata; timestamp interpretation is external). */
typedef struct bcir_tf_header {
  uint16_t version;
  uint16_t flags;
  uint32_t seq;
  uint64_t timestamp;
  uint16_t n_records;
} bcir_tf_header;

typedef enum bcir_tf_status {
  BCIR_TF_OK = 0,
  BCIR_TF_ERR_TRUNCATED = 1,   /* buffer too short for the header / claimed records / CRC */
  BCIR_TF_ERR_MAGIC = 2,       /* bytes 0..3 are not "BTLM" */
  BCIR_TF_ERR_VERSION = 3,     /* version outside [1, BCIR_TELEMETRY_FRAME_VERSION] */
  BCIR_TF_ERR_CRC = 4,         /* CRC trailer does not match the body */
  BCIR_TF_ERR_NOSPACE = 5,     /* encode: the caller `out` buffer was too small */
  BCIR_TF_ERR_FLAGS = 6,       /* reserved v1 flags were non-zero */
  BCIR_TF_ERR_TRAILING = 7,    /* exact decode: bytes remain after the one frame */
  BCIR_TF_ERR_RANGE = 8        /* requested record index is outside n_records */
} bcir_tf_status;

/* zlib-compatible CRC-32 (reflected, poly 0xEDB88320) -- DEFINED in bcir_runtime.c and
 * REUSED here (never reimplemented), so the C and Python (zlib.crc32) rails agree. */
uint32_t bcir_crc32(const uint8_t *BCIR_TF_RESTRICT data, size_t len);

/* Checked size calculation for one frame. Returns 1 and writes `*out` when the
 * header + body + CRC size is representable by this target's size_t; otherwise 0.
 * This matters on small freestanding targets because n_records is a u16 while the
 * maximum body is larger than a 16-bit address space. */
static inline int bcir_tf_frame_size_checked(uint16_t n, size_t *out) {
  const size_t fixed = (size_t)BCIR_TELEMETRY_FRAME_HEADER_SIZE
                     + (size_t)BCIR_TELEMETRY_FRAME_CRC_SIZE;
  if (!out || (size_t)n > (SIZE_MAX - fixed) / (size_t)BCIR_TF_RECORD_SIZE)
    return 0;
  *out = fixed + (size_t)n * (size_t)BCIR_TF_RECORD_SIZE;
  return 1;
}

/* Compatibility value form. SIZE_MAX is the explicit unrepresentable sentinel;
 * codec implementations use the checked form directly. */
static inline size_t bcir_tf_frame_size(uint16_t n) {
  size_t out = 0;
  return bcir_tf_frame_size_checked(n, &out) ? out : SIZE_MAX;
}

/* Encode ONE telemetry frame into `out` (capacity `out_len`). Fields are written
 * explicitly little-endian byte by byte (host-endian-independent; no struct padding on
 * the wire) and sealed with bcir_crc32 over every preceding byte. Returns the number of
 * bytes written, or 0 on overflow (the caller buffer was too small -- the NOSPACE
 * contract). This is the bare-metal producer's egress: `out` is a byte buffer; wiring
 * it to a real byte sink belongs to the future resident-driver adapter. */
BCIR_TF_NODISCARD size_t bcir_tf_encode_frame(const bcir_tf_record *BCIR_TF_RESTRICT recs,
                                              uint16_t n, uint32_t seq, uint64_t ts,
                                              uint8_t *BCIR_TF_RESTRICT out, size_t out_len);

/* Decode + validate ONE frame at `buf[0]` (magic / version / length / CRC). On
 * BCIR_TF_OK: `*hdr` (if non-NULL) is the decoded header and `*frame_len` (if non-NULL)
 * is the total frame size consumed. Records are decoded lazily via bcir_tf_get_record
 * (zero-copy: it reads straight out of `buf`). Bounds-checked end to end -- a hostile /
 * truncated buffer returns a status, never an out-of-bounds read. This prefix decoder
 * intentionally permits following stream bytes and reports the first frame in
 * `*frame_len`; use bcir_tf_decode_exact for a strict one-frame buffer. */
BCIR_TF_NODISCARD bcir_tf_status bcir_tf_decode_frame(const uint8_t *BCIR_TF_RESTRICT buf,
                                                      size_t len, bcir_tf_header *hdr,
                                                      size_t *frame_len);

/* Strict one-frame decode: the same validation as bcir_tf_decode_frame, plus a
 * BCIR_TF_ERR_TRAILING result unless the validated frame consumes exactly `len`. */
BCIR_TF_NODISCARD bcir_tf_status bcir_tf_decode_exact(const uint8_t *BCIR_TF_RESTRICT buf,
                                                      size_t len, bcir_tf_header *hdr);

/* Decode record `i` of a validated frame straight out of `buf` (little-endian byte
 * reads). The caller must have validated the frame first (bcir_tf_decode_frame == OK)
 * This function independently checks the declared n_records bound so a hostile caller
 * cannot use a larger containing buffer to read beyond the frame body. Returns
 * BCIR_TF_OK / BCIR_TF_ERR_TRUNCATED / BCIR_TF_ERR_RANGE. */
BCIR_TF_NODISCARD bcir_tf_status bcir_tf_get_record(const uint8_t *BCIR_TF_RESTRICT buf,
                                                    size_t len, uint16_t i,
                                                    bcir_tf_record *out);

/* Resync helper: scan forward from `start` for the next "BTLM" magic. Returns the
 * offset of the next magic, or `len` if none is found (so a decoder can join mid-stream
 * / skip a corrupt frame). Bounds-safe. */
size_t bcir_tf_find_magic(const uint8_t *BCIR_TF_RESTRICT buf, size_t len, size_t start);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BCIR_TELEMETRY_FRAME_H */
