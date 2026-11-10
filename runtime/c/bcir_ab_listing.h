/*===- bcir_ab_listing.h - BCAB listing and hexadecimal dump (MC1) ------------===
 *
 * The C twin of bcir/abi/artifact_tool.py's `list` and `hexdump` (PORT-1, the BCAB half): the
 * validated directory listing (`format_listing`) and the wire-span-delimited dump
 * (`format_hexdump`), octet for octet. bcir_ab_open validates the whole bundle -- layout, both CRCs,
 * the embedded SHA-256, every entry and payload -- before the first octet reaches the sink, so a
 * bundle the oracle refuses prints nothing. Freestanding, no heap; output goes through the caller's
 * sink in pieces of at most BCIR_LISTING_CHUNK octets (bcir_listing_text.h).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_AB_LISTING_H
#define BCIR_AB_LISTING_H

#include "bcir_artifact_bundle.h"
#include "bcir_listing_text.h"

#ifdef __cplusplus
extern "C" {
#endif

/* format_listing(data), octet for octet, to `sink`. Returns bcir_ab_open's refusal, BCIR_AB_ERR_INVALID
 * for no sink, BCIR_AB_ERR_OUTPUT when the sink refused, BCIR_AB_OK when the listing was taken. */
BCIR_NODISCARD bcir_ab_status bcir_ab_format_listing(const uint8_t *data, size_t len,
                                                     bcir_sink_fn sink, void *ctx);

/* format_hexdump(data, width=width), octet for octet, to `sink`; a width outside [1, 64] is
 * BCIR_AB_ERR_INVALID, checked before the bundle. Otherwise as bcir_ab_format_listing. */
BCIR_NODISCARD bcir_ab_status bcir_ab_format_hexdump(const uint8_t *data, size_t len,
                                                     uint32_t width, bcir_sink_fn sink, void *ctx);

/* The payload views `dis` and `mnemonics` give without a resident disassembler (disassemble_variant,
 * format_mnemonics), for a payload a validated bundle holds:
 *   bcir_ab_text_listing       _text_listing(payload): the text, with "\n" added when it lacks one;
 *   bcir_ab_text_mnemonics     the first word of every line that is not empty and not a comment (#, ;,
 *                              //, . after the line's edge whitespace), each followed by "\n" -- lines,
 *                              edges and words as str.splitlines(), str.strip() and str.split() find
 *                              them (Python's fixed sets of line breaks and whitespace);
 *   bcir_ab_pack_mnemonics     every segment's opcode, each followed by "\n" ("\n" alone for none).
 * The text calls take UTF-8 (BCAB admits no other text payload); the pack call checks the pack as the
 * StreamPack walks do and returns BCIR_AB_ERR_PAYLOAD for one they refuse. */
BCIR_NODISCARD bcir_ab_status bcir_ab_text_listing(const uint8_t *text, size_t len,
                                                   bcir_sink_fn sink, void *ctx);
BCIR_NODISCARD bcir_ab_status bcir_ab_text_mnemonics(const uint8_t *text, size_t len,
                                                     bcir_sink_fn sink, void *ctx);
BCIR_NODISCARD bcir_ab_status bcir_ab_pack_mnemonics(const uint8_t *pack, size_t len,
                                                     bcir_sink_fn sink, void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_AB_LISTING_H */
