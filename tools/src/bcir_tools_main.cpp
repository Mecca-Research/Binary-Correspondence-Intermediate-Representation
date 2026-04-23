#include <iostream>

#include "bcir/dialect.hpp"
#include "bcir/lowering.hpp"
#include "bcir/runtime.hpp"

int main() {
  std::cout << bcir::dialect_component_banner() << '\n';
  std::cout << bcir::lowering_component_banner() << '\n';
  std::cout << bcir::runtime_component_banner() << '\n';
  return 0;
}
