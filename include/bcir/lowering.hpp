#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
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

enum class LlvmTargetBackend { Neutral, X86, ARM, GPU, WASM };

struct RopToLlvmLowering {
  std::string opcode;
  std::string category;
  std::string neutralRoutine;
  std::string irTemplate;
  std::map<LlvmTargetBackend, std::string> backendExtensionPoints;
  bool isAtomic = false;
  bool isBarrier = false;
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

// Returns the canonical, target-neutral ROP opcode -> LLVM lowering table with
// backend-specific extension hooks for x86/ARM/GPU/WASM.
std::vector<RopToLlvmLowering> rop_opcode_to_llvm_lowerings();

// Looks up a single opcode entry from rop_opcode_to_llvm_lowerings().
const RopToLlvmLowering* find_rop_opcode_lowering(std::string_view opcode);

// Produces a textual IR-dispatch sketch for the requested instruction/backend.
std::string lower_rop_instruction_to_llvm_dispatch(
    const RopInstruction& instruction,
    LlvmTargetBackend backend = LlvmTargetBackend::Neutral);

}  // namespace bcir
