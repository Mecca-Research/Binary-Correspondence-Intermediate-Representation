#pragma once

#include <string>
#include <vector>

#include "bcir/bcir_ir.hpp"
#include "bcir/dialect.hpp"

namespace bcir {

BcirGraph build_core_graph(const BlockNode& block, std::vector<Diagnostic>* diagnostics);
bool verify_epoch_phase(const BcirGraph& graph, std::vector<Diagnostic>* diagnostics);
bool verify_registry_descriptors(const BcirGraph& graph, std::vector<Diagnostic>* diagnostics);
bool verify_hazards(const BcirGraph& graph, std::vector<Diagnostic>* diagnostics);
BcirSchedule build_schedule(const BcirGraph& graph);
struct BcirCoreBuildResult {
  BcirGraph graph;
  std::vector<Diagnostic> diagnostics;
};

BcirCoreBuildResult build_core_graph(const BlockNode& block);

}  // namespace bcir
