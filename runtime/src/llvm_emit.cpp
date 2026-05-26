#include "bcir/llvm_emit.hpp"

#include <sstream>

namespace bcir {

std::string emit_textual_llvm_ir(const BcirGraph& graph, const BcirSchedule& schedule,
                                 const std::string& module_name) {
  std::ostringstream out;
  out << "; ModuleID = '" << module_name << "'\n";
  out << "source_filename = \"" << module_name << "\"\n\n";
  out << "define void @bcir_entry(ptr %mem0) {\nentry:\n";
  std::size_t tmp = 0;
  for (auto node_id : schedule.executionOrder) {
    const auto& n = graph.nodes[node_id];
    out << "  ; !bcir.node !{" << n.id << "} !bcir.phase !{" << n.phase << "}\n";
    switch (n.opcode) {
      case BcirOpcode::Load: out << "  %t" << tmp++ << " = load i32, ptr %mem0, align 4\n"; break;
      case BcirOpcode::Store: out << "  store i32 0, ptr %mem0, align 4\n"; break;
      case BcirOpcode::Add: out << "  %t" << tmp++ << " = add i32 1, 2\n"; break;
      case BcirOpcode::Sub: out << "  %t" << tmp++ << " = sub i32 3, 1\n"; break;
      case BcirOpcode::Mul: out << "  %t" << tmp++ << " = mul i32 2, 4\n"; break;
      case BcirOpcode::AtomicAdd: out << "  %t" << tmp++ << " = atomicrmw add ptr %mem0, i32 1 seq_cst\n"; break;
      case BcirOpcode::AtomicSub: out << "  %t" << tmp++ << " = atomicrmw sub ptr %mem0, i32 1 seq_cst\n"; break;
      case BcirOpcode::AtomicXor: out << "  %t" << tmp++ << " = atomicrmw xor ptr %mem0, i32 1 seq_cst\n"; break;
      case BcirOpcode::CmpXchg: out << "  %t" << tmp++ << " = cmpxchg ptr %mem0, i32 0, i32 1 seq_cst seq_cst\n"; break;
      case BcirOpcode::Barrier: out << "  fence seq_cst\n"; break;
      case BcirOpcode::Phase: out << "  call void @bcir_phase(i32 " << n.phase << ")\n"; break;
      default: break;
    }
  }
  out << "  ret void\n}\n\ndeclare void @bcir_phase(i32)\n";
  out << "!bcir.module = !{!0}\n!0 = !{\"pure-llvm\"}\n";
  return out.str();
}

}  // namespace bcir
