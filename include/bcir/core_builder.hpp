#pragma once

#include <string>
#include <vector>

#include "bcir/bcir_ir.hpp"
#include "bcir/dialect.hpp"

namespace bcir {

struct BcirCoreBuildResult {
  BcirGraph graph;
  std::vector<Diagnostic> diagnostics;
};

BcirCoreBuildResult build_core_graph(const BlockNode& block);

}  // namespace bcir
