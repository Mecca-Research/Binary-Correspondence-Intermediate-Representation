#pragma once

#include <string>
#include <vector>

namespace bcir {

// Build task list for bringing up the BCIR MLIR dialect and LLVM lowering.
std::vector<std::string> bcir_mlir_build_tasks();

// Returns a legal LLVM textual module substrate used as the BCIR ABI contract.
std::string bcir_reference_llvm_abi_module();

}  // namespace bcir
