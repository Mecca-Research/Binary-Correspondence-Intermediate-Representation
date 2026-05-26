#include "bcir/llvm_emit.hpp"

#include <algorithm>
#include <sstream>

namespace bcir {

std::vector<BcirVerifierMessage> verify_epoch_phase(const BcirGraph& graph) {
  std::vector<BcirVerifierMessage> out;
  for (std::size_t i = 1; i < graph.nodes.size(); ++i) {
    const auto& prev = graph.nodes[i - 1];
    const auto& cur = graph.nodes[i];
    if (cur.epoch < prev.epoch || (cur.epoch == prev.epoch && cur.phase < prev.phase)) {
      out.push_back({false, "(epoch,phase) monotonicity violation"});
    }
    if (cur.phase < prev.phase && cur.epoch == prev.epoch) out.push_back({false, "phase reset requires epoch increment"});
  }
  if (out.empty()) out.push_back({true, "ok"});
  return out;
}
std::vector<BcirVerifierMessage> verify_registry_descriptors(const BcirGraph& graph) {
  std::vector<BcirVerifierMessage> out;
  for (const auto& n : graph.nodes) {
    if (!n.registry) continue;
    auto it = graph.registries.find(*n.registry);
    if (it == graph.registries.end()) { out.push_back({false, "missing registry descriptor"}); continue; }
    if (it->second.alignmentBytes == 0 || (n.registryOffset % it->second.alignmentBytes != 0)) out.push_back({false, "alignment violation"});
    if (n.registryOffset >= it->second.capacityBytes) out.push_back({false, "capacity violation"});
    if ((n.opcode == BcirOpcode::AtomicAdd || n.opcode == BcirOpcode::AtomicSub || n.opcode == BcirOpcode::AtomicXor) && !it->second.supportsAtomic) out.push_back({false, "atomic capability violation"});
  }
  if (out.empty()) out.push_back({true, "ok"});
  return out;
}
std::vector<BcirVerifierMessage> verify_hazards(const BcirGraph& graph) {
  std::vector<BcirVerifierMessage> out;
  for (const auto& e : graph.edges) if (e.kind == BcirEdgeKind::HazardOrder && e.src == e.dst) out.push_back({false, "self hazard edge"});
  if (out.empty()) out.push_back({true, "ok"});
  return out;
}
BcirSchedule build_schedule(const BcirGraph& graph) {
  std::vector<const BcirNode*> order;
  for (const auto& n : graph.nodes) order.push_back(&n);
  std::sort(order.begin(), order.end(), [](const BcirNode* a, const BcirNode* b) {
    if (a->epoch != b->epoch) return a->epoch < b->epoch;
    if (a->phase != b->phase) return a->phase < b->phase;
    if (a->lane != b->lane) return a->lane < b->lane;
    if (a->opcode != b->opcode) return a->opcode < b->opcode;
    if (a->hazardDomain != b->hazardDomain) return a->hazardDomain < b->hazardDomain;
    return a->id < b->id;
  });
  BcirSchedule s;
  std::uint64_t key = UINT64_MAX;
  for (std::size_t i = 0; i < order.size(); ++i) {
    const auto* n = order[i];
    s.executionOrder.push_back(n->id);
    std::uint64_t next = (static_cast<std::uint64_t>(n->epoch) << 32u) | n->phase;
    if (next != key) { if (!s.phaseEnd.empty()) s.phaseEnd.back() = i; s.phaseBegin.push_back(i); s.phaseEnd.push_back(i); key = next; }
  }
  if (!s.phaseEnd.empty()) s.phaseEnd.back() = s.executionOrder.size();
  return s;
}

std::string emit_textual_llvm_ir(const BcirGraph& graph, const BcirSchedule& schedule, const std::string& fn_name) {
  std::ostringstream os;
  os << "; BCIR textual LLVM IR backend\n";
  os << "define void @" << fn_name << "(ptr %base) {\nentry:\n";
  for (auto id : schedule.executionOrder) {
    const auto& n = graph.nodes.at(static_cast<std::size_t>(id));
    os << "  ; !bcir.node !{" << n.id << ", " << n.epoch << ", " << n.phase << "}\n";
    switch (n.opcode) {
      case BcirOpcode::Load: os << "  %n" << n.id << " = load i32, ptr %base, align 4\n"; break;
      case BcirOpcode::Store: os << "  store i32 0, ptr %base, align 4\n"; break;
      case BcirOpcode::Add: os << "  %n" << n.id << " = add i32 1, 2\n"; break;
      case BcirOpcode::AtomicAdd: os << "  %n" << n.id << " = atomicrmw add ptr %base, i32 1 seq_cst, align 4\n"; break;
      case BcirOpcode::CmpXchg: os << "  %n" << n.id << " = cmpxchg ptr %base, i32 0, i32 1 seq_cst monotonic, align 4\n"; break;
      case BcirOpcode::Barrier: os << "  fence seq_cst\n"; break;
      default: os << "  ; noop\n"; break;
    }
  }
  os << "  ret void\n}\n";
  return os.str();
}

}  // namespace bcir
