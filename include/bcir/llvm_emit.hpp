#pragma once

#include <string>
#include <vector>

#include "bcir/bcir_ir.hpp"

namespace bcir {

std::string emit_textual_llvm_ir(const BcirGraph& graph, const BcirSchedule& schedule,
                                 const std::string& module_name = "bcir.module");

}
struct BcirVerifierMessage {
  bool ok = true;
  std::string message;
};

std::vector<BcirVerifierMessage> verify_epoch_phase(const BcirGraph& graph);
std::vector<BcirVerifierMessage> verify_registry_descriptors(const BcirGraph& graph);
std::vector<BcirVerifierMessage> verify_hazards(const BcirGraph& graph);
BcirSchedule build_schedule(const BcirGraph& graph);
std::string emit_textual_llvm_ir(const BcirGraph& graph, const BcirSchedule& schedule, const std::string& fn_name);

}  // namespace bcir
