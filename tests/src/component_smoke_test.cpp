#include <cstdlib>
#include <iostream>
#include <string>

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
  }

  fn main() {
    ld rid=r0 lane=lane2 from mem0;
    st rid=r1 lane=lane5 to mem1;
    lane shuffle r1 lane7;
    phase 2;
    barrier workgroup;
    expand pair(r0, r1);
  }
}
)";

  const bcir::ParseResult parsed_valid = bcir::parse_dialect(valid_program);
  if (!parsed_valid.diagnostics.empty()) {
    std::cerr << "Expected valid program to parse without diagnostics" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string invalid_program = R"(
module test {
  macro broken(a, a) {
    phase 9;
  }

  fn main() {
    ld rid=rx lane=laneZZ from mem0;
    lane rotate x lane77;
  }
}
)";

  const bcir::ParseResult parsed_invalid = bcir::parse_dialect(invalid_program);
  if (!has_diagnostic_substring(parsed_invalid, "malformed RID token") ||
      !has_diagnostic_substring(parsed_invalid, "malformed lane token") ||
      !has_diagnostic_substring(parsed_invalid, "malformed macro params") ||
      !has_diagnostic_substring(parsed_invalid, "invalid phase directive") ||
      !has_diagnostic_substring(parsed_invalid, "invalid lane directive")) {
    std::cerr << "Expected parser diagnostics were not emitted" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "BCIR smoke test passed" << std::endl;
  return EXIT_SUCCESS;
}
