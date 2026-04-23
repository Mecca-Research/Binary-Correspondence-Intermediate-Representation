#include <cstdlib>
#include <iostream>
#include <string>

#include "bcir/dialect.hpp"
#include "bcir/lowering.hpp"
#include "bcir/runtime.hpp"

int main() {
  const std::string dialect = bcir::dialect_component_banner();
  const std::string lowering = bcir::lowering_component_banner();
  const std::string runtime = bcir::runtime_component_banner();

  const bool ok = !dialect.empty() && !lowering.empty() && !runtime.empty();
  if (!ok) {
    std::cerr << "BCIR component banners must not be empty" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "BCIR smoke test passed" << std::endl;
  return EXIT_SUCCESS;
}
