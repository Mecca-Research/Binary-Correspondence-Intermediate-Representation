#include "bcir_artifact_bundle.hpp"

#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  std::FILE *stream = std::fopen(argv[1], "rb");
  if (!stream)
    return 2;
  if (std::fseek(stream, 0, SEEK_END) != 0) {
    std::fclose(stream);
    return 2;
  }
  long length = std::ftell(stream);
  if (length < 0 || length > 64L * 1024L * 1024L || std::fseek(stream, 0, SEEK_SET) != 0) {
    std::fclose(stream);
    return 2;
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
  if (std::fread(bytes.data(), 1, bytes.size(), stream) != bytes.size()) {
    std::fclose(stream);
    return 2;
  }
  std::fclose(stream);
  try {
    bcir::ArtifactBundleView bundle(bytes.data(), bytes.size());
    if (bundle.count() != 3 || bundle.select_default().id() != "portable-c")
      return 1;
    bcir::ArtifactEnvelope envelope;
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
    auto selected = bundle.select(envelope);
    if (selected.id() != "x86-avx2" || selected.size() != 20)
      return 1;
    std::printf("OK C++ entries=%u selected=%s\n", bundle.count(), selected.id().c_str());
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  return 0;
}
