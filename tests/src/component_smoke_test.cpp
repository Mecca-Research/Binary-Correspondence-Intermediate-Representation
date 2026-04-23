#include <cstdlib>
#include <iostream>
#include <string>
#include <algorithm>

#include "bcir/dialect.hpp"
#include "bcir/lowering.hpp"
#include "bcir/runtime.hpp"

namespace {

bool has_diagnostic_substring(const bcir::ParseResult& result,
                              const std::string& needle) {
  for (const auto& diagnostic : result.diagnostics) {
    if (diagnostic.message.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool has_diagnostic_substring(const bcir::VerifyResult& result,
                              const std::string& needle) {
  for (const auto& diagnostic : result.diagnostics) {
    if (diagnostic.message.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

}  // namespace

int main() {
  const std::string dialect = bcir::dialect_component_banner();
  const std::string lowering = bcir::lowering_component_banner();
  const std::string runtime = bcir::runtime_component_banner();

  const bool banners_ok = !dialect.empty() && !lowering.empty() && !runtime.empty();
  if (!banners_ok) {
    std::cerr << "BCIR component banners must not be empty" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string valid_program = R"(
module test {
  macro pair(x, y) {
    bin add r3, x, y;
    map_load rid=r3 lane=lane2 from mem8;
  }

  fn main() {
    ld rid=r0<U x 8 x i32> lane=lane2 from mem0;
    ld rid=r1<U x 8 x i32> lane=lane5 from mem1;
    st rid=r1<U x 8 x i32> lane=lane5 to mem1;
    bin add r2<U x 8 x i32>, r0<U x 8 x i32>, r1<U x 8 x i32>;
    lane shuffle r1 lane7;
    phase 2;
    barrier workgroup;
    expand pair(r0, r1);
    map_store rid=r2 lane=lane2 to mem8;
    map_atomic_add rid=r2 lane=lane2 from mem8;
  }
}
)";

  bcir::ParseResult parsed_valid = bcir::parse_dialect(valid_program);
  if (!parsed_valid.diagnostics.empty()) {
    std::cerr << "Expected valid program to parse without diagnostics" << std::endl;
    return EXIT_FAILURE;
  }
  std::vector<bcir::Diagnostic> expand_diags =
      bcir::expand_macros(&parsed_valid.module);
  if (!expand_diags.empty()) {
    std::cerr << "Expected macro expansion to succeed for valid program" << std::endl;
    return EXIT_FAILURE;
  }
  bcir::lower_map_surface_ops(&parsed_valid.module);
  const auto& lowered_ops = parsed_valid.module.functions.front().body.operations;
  if (std::any_of(lowered_ops.begin(), lowered_ops.end(), [](const auto& op) {
        return op->kind == bcir::Operation::Kind::MapSurface ||
               op->kind == bcir::Operation::Kind::MacroExpansion;
      })) {
    std::cerr << "Expected MAP and macro surface operations to be lowered away"
              << std::endl;
    return EXIT_FAILURE;
  }

  const bcir::VerifyResult verified_valid = bcir::verify_rop(parsed_valid.module);
  if (!verified_valid.ok || !verified_valid.diagnostics.empty() ||
      verified_valid.passes.size() != 4) {
    std::cerr << "Expected valid program to verify without diagnostics"
              << std::endl;
    return EXIT_FAILURE;
  }

  const std::string invalid_program = R"(
module test {
  macro broken(a, a) {
    phase 9;
  }

  fn main() {
    ld rid=rx<Z x 8 x i32> lane=laneZZ from mem0;
    ld rid=r0<U x 0 x i32> lane=lane1 from mem2;
    ld rid=r1<U x 8 x nope> lane=lane2 from mem3;
    ld rid=r2<UX x 4 x i16> lane=lane3 from mem4;
    ld rid=r3<T x 4 x i16> lane=lane4 from mem5;
    bin add r4<UX x 4 x i16>, r2<UX x 4 x i16>, r3<T x 4 x i16>;
    st rid=r3<T x 4 x i16> lane=lane4 to mem4;
    lane rotate x lane77;
  }
}
)";

  const bcir::ParseResult parsed_invalid = bcir::parse_dialect(invalid_program);
  if (!has_diagnostic_substring(parsed_invalid, "malformed RID token") ||
      !has_diagnostic_substring(parsed_invalid, "malformed lane token") ||
      !has_diagnostic_substring(parsed_invalid, "malformed macro params") ||
      !has_diagnostic_substring(parsed_invalid, "invalid phase directive") ||
      !has_diagnostic_substring(parsed_invalid, "invalid lane directive") ||
      !has_diagnostic_substring(parsed_invalid, "invalid lane enum") ||
      !has_diagnostic_substring(parsed_invalid, "invalid width constraint") ||
      !has_diagnostic_substring(parsed_invalid, "invalid data type") ||
      !has_diagnostic_substring(parsed_invalid, "invalid cross-lane arithmetic") ||
      !has_diagnostic_substring(parsed_invalid, "type mismatch")) {
    std::cerr << "Expected parser diagnostics were not emitted" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string invalid_verify_program = R"(
module test {
  fn main() {
    ld rid=r0<U x 8 x i32> lane=lane3 from mem0;
    st rid=r0<U x 8 x i32> lane=lane4 to mem0;
    lane shuffle r0 lane5;
    phase 1;
    phase 0;
  }
}
)";
  const bcir::ParseResult parsed_invalid_verify =
      bcir::parse_dialect(invalid_verify_program);
  if (!parsed_invalid_verify.diagnostics.empty()) {
    std::cerr << "Expected verifier-negative program to parse successfully"
              << std::endl;
    return EXIT_FAILURE;
  }
  const bcir::VerifyResult verified_invalid =
      bcir::verify_rop(parsed_invalid_verify.module);
  if (verified_invalid.ok ||
      !has_diagnostic_substring(verified_invalid, "multiple lanes") ||
      !has_diagnostic_substring(verified_invalid, "monotonic") ||
      !has_diagnostic_substring(verified_invalid, "illegal for lane parity") ||
      !has_diagnostic_substring(verified_invalid, "requires barrier")) {
    std::cerr << "Expected verifier diagnostics were not emitted" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string bad_macro_program = R"(
module test {
  macro pair(x, y) {
    bin add r1, x, y;
  }
  fn main() {
    expand pair(r0);
  }
}
)";
  bcir::ParseResult parsed_bad_macro = bcir::parse_dialect(bad_macro_program);
  std::vector<bcir::Diagnostic> bad_macro_diags =
      bcir::expand_macros(&parsed_bad_macro.module);
  if (!has_diagnostic_substring(parsed_bad_macro, "expected macro argument") &&
      !has_diagnostic_substring(bcir::VerifyResult{true, {}, bad_macro_diags},
                                "expects 2 argument")) {
    std::cerr << "Expected macro expansion diagnostics were not emitted"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "BCIR smoke test passed" << std::endl;
  return EXIT_SUCCESS;
}
