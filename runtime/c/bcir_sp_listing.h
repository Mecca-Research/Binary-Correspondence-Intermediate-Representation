/*===- bcir_sp_listing.h - StreamPack listing and hexadecimal dump (MC1) ------===
 *
 * The C twin of bcir/abi/streampack_tool.py (PORT-1, docs/research/BCIR_NATIVE_PORT_AUDIT_2026-11-09.md
 * N1): the validated, provenance-aware assembly listing (`format_listing`) and the record-delimited
 * hexadecimal/ASCII dump (`format_hexdump`), octet for octet. The record boundaries are the ones
 * bcir_sp_for_each_record finds with the runtime's own cursor, so no second wire parser exists here,
 * and a pack the oracle's decoder refuses prints nothing: the whole pack is checked before the first
 * octet reaches the sink.
 *
 * Freestanding, no heap: output goes through the caller's sink in pieces of at most
 * BCIR_LISTING_CHUNK octets, so a listing of any length needs no buffer of that length. The listing
 * borrows scratch for its trace index (bcir_sp_listing_scratch_bytes says how much); the dump
 * borrows none.
 *
 * Quoted strings are spelled as Python's ascii() spells a str: repr's quote choice and escapes, with
 * every code point past U+007E escaped (\xhh, \uhhhh, \Uhhhhhhhh). repr itself keeps a printable
 * non-ASCII character literal, and which characters are printable is the interpreter's Unicode
 * database -- a listing that changed with the Python that printed it would not be deterministic.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_SP_LISTING_H
#define BCIR_SP_LISTING_H

#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The most octets one sink call carries. */
#define BCIR_LISTING_CHUNK 1024u

/* The widest hexadecimal-dump row (format_hexdump's [1, 64]). */
#define BCIR_HEXDUMP_WIDTH_MAX 64u

/* Take `len` octets of output; return 0 when taken, nonzero to refuse (the format call then stops
 * and returns BCIR_ERR_NOSPACE). */
typedef int (*bcir_sink_fn)(void *ctx, const char *text, size_t len);

/* The scratch octets bcir_sp_format_listing borrows for `data`: 12 per trace record. The whole pack
 * is checked first, so the answer is never more than half its length; a refused pack returns its
 * status and leaves *need at zero. */
BCIR_NODISCARD bcir_status bcir_sp_listing_scratch_bytes(const uint8_t *BCIR_RESTRICT data,
                                                         size_t len, size_t *BCIR_RESTRICT need);

/* format_listing(data), octet for octet, to `sink`. `scratch[0..scratch_len)` holds the trace index
 * (at least bcir_sp_listing_scratch_bytes; NULL with zero when the pack has no trace records); it must
 * not overlap `data`. A segment whose claim has several trace records shows the last, as the oracle's
 * claim-to-note map does. Returns the pack's refusal status, BCIR_ERR_NOSPACE for short scratch or a
 * sink that refused, BCIR_OK when the whole listing was taken. */
BCIR_NODISCARD bcir_status bcir_sp_format_listing(const uint8_t *BCIR_RESTRICT data, size_t len,
                                                  uint8_t *BCIR_RESTRICT scratch, size_t scratch_len,
                                                  bcir_sink_fn sink, void *ctx);

/* format_hexdump(data, width=width), octet for octet, to `sink`. A width outside
 * [1, BCIR_HEXDUMP_WIDTH_MAX] is BCIR_ERR_OVERFLOW (no row of that width is formatted), checked
 * before the pack. Otherwise as bcir_sp_format_listing. */
BCIR_NODISCARD bcir_status bcir_sp_format_hexdump(const uint8_t *BCIR_RESTRICT data, size_t len,
                                                  uint32_t width, bcir_sink_fn sink, void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_SP_LISTING_H */
