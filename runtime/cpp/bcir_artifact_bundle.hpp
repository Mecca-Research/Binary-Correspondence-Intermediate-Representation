// RAII/error-mapping wrapper for the allocation-free C BCAB reader.
#ifndef BCIR_ARTIFACT_BUNDLE_HPP
#define BCIR_ARTIFACT_BUNDLE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" {
#include "bcir_artifact_bundle.h"
}

namespace bcir {

class ArtifactBundleError : public std::runtime_error {
public:
  explicit ArtifactBundleError(bcir_ab_status status)
      : std::runtime_error(std::string("BCAB: ") + bcir_ab_status_string(status)), status_(status) {
  }
  bcir_ab_status status() const noexcept { return status_; }

private:
  bcir_ab_status status_;
};

struct ArtifactEnvelope {
  std::string triple;
  std::string architecture;
  std::string os_abi;
  std::string channel;
  std::string features;
  std::uint64_t accepted_kind_mask = 0;
  std::uint64_t accepted_format_mask = 0;
  bcir_ab_endianness endianness = BCIR_AB_ENDIAN_NEUTRAL;
  std::uint8_t pointer_bits = 0;
  std::uint32_t machine = 0;
  std::optional<std::array<std::uint8_t, 32>> target_manifest_sha256;
  std::optional<std::uint64_t> cal_gen;
  bool require_r12 = true;
  bool allow_debug = false;

  bcir_ab_envelope c_view() const noexcept {
    bcir_ab_envelope output{};
    output.triple = triple.empty() ? nullptr : triple.c_str();
    output.architecture = architecture.empty() ? nullptr : architecture.c_str();
    output.os_abi = os_abi.empty() ? nullptr : os_abi.c_str();
    output.channel = channel.empty() ? nullptr : channel.c_str();
    output.features = features.empty() ? nullptr : features.c_str();
    output.accepted_kind_mask = accepted_kind_mask;
    output.accepted_format_mask = accepted_format_mask;
    output.endianness = static_cast<std::uint8_t>(endianness);
    output.pointer_bits = pointer_bits;
    output.machine = machine;
    output.target_manifest_sha256 =
        target_manifest_sha256 ? target_manifest_sha256->data() : nullptr;
    output.cal_gen = cal_gen.value_or(0);
    output.has_cal_gen = cal_gen.has_value() ? 1 : 0;
    output.require_r12 = require_r12 ? 1 : 0;
    output.allow_debug = allow_debug ? 1 : 0;
    return output;
  }
};

class ArtifactVariantView {
public:
  explicit ArtifactVariantView(bcir_ab_entry entry) : entry_(entry) {}
  std::string id() const { return entry_.variant_id; }
  bcir_ab_kind kind() const { return static_cast<bcir_ab_kind>(entry_.kind); }
  bcir_ab_format format() const { return static_cast<bcir_ab_format>(entry_.format); }
  const std::uint8_t *data() const noexcept { return entry_.payload; }
  std::size_t size() const noexcept { return static_cast<std::size_t>(entry_.payload_size); }
  const bcir_ab_entry &metadata() const noexcept { return entry_; }

private:
  bcir_ab_entry entry_{};
};

class ArtifactBundleView {
public:
  // BORROWED bytes: owner must outlive this view and every ArtifactVariantView.
  ArtifactBundleView(const std::uint8_t *data, std::size_t size) {
    bcir_ab_status status = bcir_ab_open(data, size, &view_);
    if (status != BCIR_AB_OK)
      throw ArtifactBundleError(status);
  }

  std::size_t size() const noexcept { return view_.length; }
  std::uint32_t count() const noexcept { return view_.count; }
  std::uint64_t generation() const noexcept { return view_.generation; }

  ArtifactVariantView at(std::uint32_t index) const {
    bcir_ab_entry entry{};
    bcir_ab_status status = bcir_ab_get(&view_, index, &entry);
    if (status != BCIR_AB_OK)
      throw ArtifactBundleError(status);
    return ArtifactVariantView(entry);
  }

  ArtifactVariantView select_default() const { return select_impl(nullptr, nullptr); }

  ArtifactVariantView select(const ArtifactEnvelope &envelope,
                             const std::string &requested_id = {}) const {
    bcir_ab_envelope c_envelope = envelope.c_view();
    return select_impl(&c_envelope, requested_id.empty() ? nullptr : requested_id.c_str());
  }

private:
  ArtifactVariantView select_impl(const bcir_ab_envelope *envelope,
                                  const char *requested_id) const {
    std::uint32_t index = BCIR_AB_NO_INDEX;
    bcir_ab_status status = bcir_ab_select(&view_, envelope, requested_id, &index);
    if (status != BCIR_AB_OK)
      throw ArtifactBundleError(status);
    return at(index);
  }

  bcir_ab_view view_{};
};

} // namespace bcir

#endif // BCIR_ARTIFACT_BUNDLE_HPP
