#pragma once

#include <string>

#include "bcir/dialect.hpp"

namespace bcir {

// BCIR <-> ROP <-> LLVM conversion interface.
std::string lowering_component_banner();

// Expands all macro invocations with parameter binding and hygienic temporary
// register allocation. Returns diagnostics for malformed invocations.
std::vector<Diagnostic> expand_macros(ModuleNode* module);

// Lowers all MAP surface operations (load/store/atomic/...) into canonical ROP
// instruction sequences.
void lower_map_surface_ops(ModuleNode* module);

}  // namespace bcir
