/* Hosted parity harness for the freestanding BCAB reader/selector. */
#include "bcir_artifact_bundle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int zero_view(const bcir_ab_view *view) {
  const unsigned char *p = (const unsigned char *)view;
  for (size_t i = 0; i < sizeof(*view); ++i) if (p[i] != 0) return 0;
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) return 2;
  FILE *stream = fopen(argv[1], "rb");
  if (!stream || fseek(stream, 0, SEEK_END) != 0) return 2;
  long length = ftell(stream);
  if (length < 0 || (unsigned long)length > 64ul * 1024ul * 1024ul ||
      fseek(stream, 0, SEEK_SET) != 0) return 2;
  unsigned char *data = (unsigned char *)malloc((size_t)length);
  if (!data || fread(data, 1, (size_t)length, stream) != (size_t)length) {
    free(data); fclose(stream); return 2;
  }
  fclose(stream);

  bcir_ab_view view;
  bcir_ab_status status = bcir_ab_open(data, (size_t)length, &view);
  if (argc == 3 && strcmp(argv[2], "reject") == 0) {
    if (status == BCIR_AB_OK) { free(data); return 1; }
    printf("REJECT %s\n", bcir_ab_status_string(status));
    free(data);
    return 0;
  }
  if (status != BCIR_AB_OK) {
    fprintf(stderr, "open:%s\n", bcir_ab_status_string(status)); free(data); return 1;
  }
  if (view.count != 3 || view.root_index == BCIR_AB_NO_INDEX ||
      view.default_index == BCIR_AB_NO_INDEX) { free(data); return 1; }
  bcir_ab_entry root, selected;
  if (bcir_ab_get(&view, view.root_index, &root) != BCIR_AB_OK ||
      root.kind != BCIR_AB_STREAM_PACK) { free(data); return 1; }
  uint32_t index = BCIR_AB_NO_INDEX;
  if (bcir_ab_select(&view, NULL, NULL, &index) != BCIR_AB_OK ||
      bcir_ab_get(&view, index, &selected) != BCIR_AB_OK ||
      strcmp(selected.variant_id, "portable-c") != 0) { free(data); return 1; }

  bcir_ab_envelope envelope;
  memset(&envelope, 0, sizeof(envelope));
  envelope.triple = "x86_64-unknown-linux-gnu";
  envelope.architecture = "x86_64";
  envelope.os_abi = "linux-gnu";
  envelope.channel = "host";
  envelope.features = "avx2";
  envelope.accepted_kind_mask = UINT64_C(1) << BCIR_AB_ELF_OBJECT;
  envelope.accepted_format_mask = UINT64_C(1) << BCIR_AB_FMT_ELF;
  envelope.endianness = BCIR_AB_ENDIAN_LITTLE;
  envelope.pointer_bits = 64;
  envelope.machine = 62;
  envelope.require_r12 = 1;
  if (bcir_ab_select(&view, &envelope, NULL, &index) != BCIR_AB_OK ||
      bcir_ab_get(&view, index, &selected) != BCIR_AB_OK ||
      strcmp(selected.variant_id, "x86-avx2") != 0) { free(data); return 1; }
  if (bcir_ab_select(&view, &envelope, "portable-c", &index) != BCIR_AB_ERR_INCOMPATIBLE) {
    free(data); return 1;
  }

  /* Caller-owned selector inputs are bounded and canonical before matching. */
  {
    char unterminated_id[BCIR_AB_VARIANT_ID_MAX + 1u];
    char unterminated_features[BCIR_AB_FEATURE_CSV_MAX + 1u];
    memset(unterminated_id, 'x', sizeof(unterminated_id));
    memset(unterminated_features, 'x', sizeof(unterminated_features));
    if (bcir_ab_select(&view, &envelope, unterminated_id, &index) != BCIR_AB_ERR_INVALID) {
      free(data); return 1;
    }
    envelope.features = unterminated_features;
    if (bcir_ab_select(&view, &envelope, NULL, &index) != BCIR_AB_ERR_INVALID) {
      free(data); return 1;
    }
    envelope.features = "avx2";
    envelope.require_r12 = 2;
    if (bcir_ab_select(&view, &envelope, NULL, &index) != BCIR_AB_ERR_INVALID) {
      free(data); return 1;
    }
    envelope.require_r12 = 1;
    envelope.accepted_kind_mask = UINT64_C(1) << 63;
    if (bcir_ab_select(&view, &envelope, NULL, &index) != BCIR_AB_ERR_INVALID) {
      free(data); return 1;
    }
    envelope.accepted_kind_mask = UINT64_C(1) << BCIR_AB_ELF_OBJECT;
  }

  /* A failing parse resets the caller's output, including borrowed pointers. */
  bcir_ab_view failed = view;
  status = bcir_ab_open(data, BCIR_AB_HEADER_SIZE - 1u, &failed);
  if (status != BCIR_AB_ERR_SIZE || !zero_view(&failed)) { free(data); return 1; }
  printf("OK entries=%u root=%s selected=%s\n", view.count, root.variant_id,
         selected.variant_id);
  free(data);
  return 0;
}
