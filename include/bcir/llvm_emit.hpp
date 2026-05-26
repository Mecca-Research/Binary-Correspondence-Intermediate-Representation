#pragma once

#include <string>
#include "bcir/bcir_ir.hpp"

namespace bcir {

std::string emit_textual_llvm_ir(const BcirGraph& graph, const BcirSchedule& schedule,
                                 const std::string& module_name = "bcir.module");

}
