/* bcir_bytes.h -- the runtime's one copy of the fixed-width integer loads and stores (CSUB-3c).
 *
 * Every wire format the runtime reads or writes spells its integers a byte at a time -- StreamPack, BCAB, BTLM, the
 * telemetry frame and envelope, the shard manifest, the control plane, the ring, the plan signature store, the Q8
 * model, token shards, JER's binary records, and SHA-256's and SHA-512's big-endian words -- so a load or a store is
 * the same on every host whatever its byte order or alignment. Fifteen units had each defined their own (`rd16`,
 * `rd32`, `rd64`, `wr16`, `wr32`, `wr64`, `ld32`, `st32`, `read_u32`, `rd32be`, `load64_be`): one mechanism, fifteen
 * copies (L14). This header is the one copy, and `test_the_runtime_has_one_copy_of_its_byte_order_helpers` refuses
 * another.
 *
 * Freestanding (<stdint.h> alone), heap-free, and in the verified C subset: each unit that includes it compiles with
 * R1-R18 clean on both C front-end rails as before (`tools/c/csub_ratchet.py`). Each function is held to Python's
 * `int.from_bytes` and `int.to_bytes` on every width and both byte orders by the same test.
 */
#ifndef BCIR_BYTES_H
#define BCIR_BYTES_H

#include <stdint.h>

/* --- little-endian --- */
static inline uint16_t bcir_ld16le(const uint8_t *p) { return (uint16_t)((uint16_t)p[0] | (uint16_t)(p[1] << 8)); }
static inline uint32_t bcir_ld32le(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static inline uint64_t bcir_ld64le(const uint8_t *p) {
  return (uint64_t)bcir_ld32le(p) | (uint64_t)bcir_ld32le(p + 4) << 32;
}
static inline void bcir_st16le(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
}
static inline void bcir_st32le(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}
static inline void bcir_st64le(uint8_t *p, uint64_t v) {
  bcir_st32le(p, (uint32_t)v);
  bcir_st32le(p + 4, (uint32_t)(v >> 32));
}

/* --- big-endian: ELF's and the digests' words --- */
static inline uint16_t bcir_ld16be(const uint8_t *p) { return (uint16_t)((uint16_t)(p[0] << 8) | (uint16_t)p[1]); }
static inline uint32_t bcir_ld32be(const uint8_t *p) {
  return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | (uint32_t)p[3];
}
static inline uint64_t bcir_ld64be(const uint8_t *p) {
  return (uint64_t)bcir_ld32be(p) << 32 | (uint64_t)bcir_ld32be(p + 4);
}
static inline void bcir_st32be(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v >> 24);
  p[1] = (uint8_t)(v >> 16);
  p[2] = (uint8_t)(v >> 8);
  p[3] = (uint8_t)v;
}

#endif /* BCIR_BYTES_H */
