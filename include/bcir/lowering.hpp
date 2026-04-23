#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "bcir/dialect.hpp"

namespace bcir {

// BCIR <-> ROP <-> LLVM conversion interface.
std::string lowering_component_banner();

struct RopDependency {
  std::size_t producerIndex = 0;
  std::string kind;
};

struct RopInstruction {
  std::string opcode;
  std::string rid;
  std::optional<std::string> laneType;
  int lane = 0;
  std::optional<int> offset;
  std::optional<int> phase;
  std::vector<RopDependency> dependencies;

  // Auxiliary metadata needed to reconstruct BCIR graph nodes losslessly.
  bool isLoad = false;
  std::string target;
  std::string lhs;
  std::string rhs;
  std::string barrierScope;
};

// Expands all macro invocations with parameter binding and hygienic temporary
// register allocation. Returns diagnostics for malformed invocations.
std::vector<Diagnostic> expand_macros(ModuleNode* module);

// Lowers all MAP surface operations (load/store/atomic/...) into canonical ROP
// instruction sequences.
void lower_map_surface_ops(ModuleNode* module);

// Translates BCIR graph nodes into a linear ROP instruction stream.
std::vector<RopInstruction> bcir_graph_to_rop_stream(const BlockNode& block);

// Reconstructs BCIR graph nodes from a linear ROP instruction stream.
BlockNode rop_stream_to_bcir_graph(const std::vector<RopInstruction>& stream);

}  // namespace bcir
