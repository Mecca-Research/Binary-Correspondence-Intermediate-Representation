/* libFuzzer harness for the allocation-free BCAB reader and selector. */
#include <stddef.h>
#include <stdint.h>

#include "bcir_artifact_bundle.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  bcir_ab_view view;
  if (bcir_ab_open(data, size, &view) != BCIR_AB_OK) return 0;
  uint64_t accumulator = view.generation + view.provenance_digest;
  for (uint32_t index = 0; index < view.count; ++index) {
    bcir_ab_entry entry;
    if (bcir_ab_get(&view, index, &entry) == BCIR_AB_OK) {
      accumulator += entry.payload_size + entry.payload_crc32;
      if (entry.payload_size) accumulator += entry.payload[entry.payload_size - 1u];
    }
  }
  bcir_ab_envelope envelope = {0};
  envelope.features = "avx2,simd128";
  envelope.endianness = BCIR_AB_ENDIAN_LITTLE;
  envelope.pointer_bits = 64;
  envelope.require_r12 = 1;
  uint32_t selected = BCIR_AB_NO_INDEX;
  (void)bcir_ab_select(&view, &envelope, NULL, &selected);
  return accumulator == UINT64_MAX && selected == 0 ? 1 : 0;
}
