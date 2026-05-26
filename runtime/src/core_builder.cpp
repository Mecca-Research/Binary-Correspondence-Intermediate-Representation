#include "bcir/core_builder.hpp"

#include <algorithm>
#include <tuple>
#include <unordered_map>

namespace bcir {
namespace {
Diagnostic diag(std::string code, std::string msg) { return {{1, 1, 0}, "core", std::move(code), std::move(msg)}; }

bool is_atomic(BcirOpcode op) {
  return op == BcirOpcode::AtomicAdd || op == BcirOpcode::AtomicSub || op == BcirOpcode::AtomicXor || op == BcirOpcode::CmpXchg;
}

BcirLane map_lane(int lane) {
  return lane == 0 ? BcirLane::U : BcirLane::UX;
}
BcirHazardMode mode_for(BcirOpcode op) {
  if (is_atomic(op)) return BcirHazardMode::Atomic;
  if (op == BcirOpcode::Barrier) return BcirHazardMode::Barriered;
  return BcirHazardMode::Relaxed;
}
BcirOpcode map_opcode(const Operation& op, const std::string* bin = nullptr) {
  if (op.kind == Operation::Kind::LdSt) return static_cast<const LdStOperation&>(op).isLoad ? BcirOpcode::Load : BcirOpcode::Store;
  if (op.kind == Operation::Kind::Barrier) return BcirOpcode::Barrier;
  if (op.kind == Operation::Kind::Phase) return BcirOpcode::Phase;
  if (op.kind == Operation::Kind::MapSurface) {
    auto k = static_cast<const MapSurfaceOperation&>(op).surfaceKind;
    if (k == MapSurfaceOperation::SurfaceKind::AtomicAdd) return BcirOpcode::AtomicAdd;
    if (k == MapSurfaceOperation::SurfaceKind::AtomicSub) return BcirOpcode::AtomicSub;
    if (k == MapSurfaceOperation::SurfaceKind::AtomicXor) return BcirOpcode::AtomicXor;
    return k == MapSurfaceOperation::SurfaceKind::Load ? BcirOpcode::Load : BcirOpcode::Store;
  }
  if (bin) { if (*bin == "add") return BcirOpcode::Add; if (*bin == "sub") return BcirOpcode::Sub; if (*bin == "mul") return BcirOpcode::Mul; }
  return BcirOpcode::Add;
}
}  // namespace

BcirGraph build_core_graph(const BlockNode& block, std::vector<Diagnostic>* diagnostics) {
  BcirGraph g;
  g.registries.push_back({"mem0", BcirLane::U, BcirValueType::I32, 4096, BcirAddressSpace::Global, 0, 4, 0, 100000, BcirRegistryFlag::Readable | BcirRegistryFlag::Writable | BcirRegistryFlag::AtomicCapable});
  std::unordered_map<std::string, std::uint64_t> def;
  std::uint32_t phase = 0;
  std::uint32_t epoch = 0;
  for (std::size_t i = 0; i < block.operations.size(); ++i) {
    const Operation& op = *block.operations[i];
    if (op.kind == Operation::Kind::Phase) {
      const int requested = static_cast<const PhaseOperation&>(op).phase;
      if (requested == 0) { ++epoch; phase = 0; } else { phase = static_cast<std::uint32_t>(requested); }
    }
    BcirNode n;
    n.id = i; n.epoch = epoch; n.phase = phase; n.rid = "mem0"; n.offsetBytes = 0;
    n.lane = BcirLane::U;

    if (op.kind == Operation::Kind::Binary) {
      const auto& b = static_cast<const BinaryOpOperation&>(op);
      n.opcode = map_opcode(op, &b.opcode);
      if (def.count(b.lhs)) n.operands.push_back(def[b.lhs]);
      if (def.count(b.rhs)) n.operands.push_back(def[b.rhs]);
      def[b.dst] = n.id;
    } else if (op.kind == Operation::Kind::LdSt) {
      const auto& ls = static_cast<const LdStOperation&>(op);
      n.opcode = map_opcode(op); n.rid = ls.target; n.lane = map_lane(ls.lane);
      if (ls.isLoad) def[ls.rid] = n.id; else if (def.count(ls.rid)) n.operands.push_back(def[ls.rid]);
    } else if (op.kind == Operation::Kind::MapSurface) {
      const auto& m = static_cast<const MapSurfaceOperation&>(op);
      n.opcode = map_opcode(op); n.rid = m.target; n.lane = map_lane(m.lane);
      if (def.count(m.rid)) n.operands.push_back(def[m.rid]);
      if (m.surfaceKind == MapSurfaceOperation::SurfaceKind::Load) def[m.rid] = n.id;
      if (is_atomic(n.opcode)) n.lane = BcirLane::A;
    } else {
      n.opcode = map_opcode(op);
    }
    n.hazardMode = mode_for(n.opcode);
    n.hazardDomain = n.rid.rfind("mmio", 0) == 0 ? BcirHazardDomain::MMIO : BcirHazardDomain::RAM;
    g.nodes.push_back(n);
  }

  for (const auto& n : g.nodes) for (auto src : n.operands) g.edges.push_back({src, n.id, BcirEdgeKind::DataFlow, BcirHazardKind::RAW});
  for (std::size_t i = 1; i < g.nodes.size(); ++i) {
    g.edges.push_back({i - 1, i, BcirEdgeKind::ControlFlow, BcirHazardKind::RAW});
    if (g.nodes[i - 1].epoch != g.nodes[i].epoch || g.nodes[i - 1].phase != g.nodes[i].phase) g.edges.push_back({i - 1, i, BcirEdgeKind::PhaseOrder, BcirHazardKind::RAW});
    if (g.nodes[i - 1].rid == g.nodes[i].rid) g.edges.push_back({i - 1, i, BcirEdgeKind::HazardOrder, BcirHazardKind::WAW});
  }
  (void)diagnostics;
  return g;
}

bool verify_epoch_phase(const BcirGraph& graph, std::vector<Diagnostic>* diagnostics) {
  for (std::size_t i = 1; i < graph.nodes.size(); ++i) {
    const auto& a = graph.nodes[i - 1]; const auto& b = graph.nodes[i];
    if (b.epoch < a.epoch) { if (diagnostics) diagnostics->push_back(diag("epoch.regress", "epoch regression")); return false; }
    if (b.epoch == a.epoch && b.phase < a.phase) { if (diagnostics) diagnostics->push_back(diag("phase.regress", "phase must be non-decreasing within epoch")); return false; }
    if (b.phase < a.phase && b.epoch != a.epoch + 1) { if (diagnostics) diagnostics->push_back(diag("phase.reset", "phase reset requires epoch increment")); return false; }
  }
  return true;
}

bool verify_registry_descriptors(const BcirGraph& graph, std::vector<Diagnostic>* diagnostics) {
  std::unordered_map<std::string, BcirRegistryDescriptor> map; for (const auto& r : graph.registries) map[r.rid] = r;
  for (const auto& n : graph.nodes) {
    auto it = map.find(n.rid);
    if (it == map.end()) { if (diagnostics) diagnostics->push_back(diag("registry.missing", n.rid)); return false; }
    const auto& reg = it->second;
    if (n.offsetBytes >= reg.capacityBytes) { if (diagnostics) diagnostics->push_back(diag("registry.oob", n.rid)); return false; }
    if (reg.alignmentBytes != 0 && (n.offsetBytes % reg.alignmentBytes != 0)) { if (diagnostics) diagnostics->push_back(diag("registry.align", n.rid)); return false; }
    if (is_atomic(n.opcode) && (static_cast<std::uint32_t>(reg.flags) & static_cast<std::uint32_t>(BcirRegistryFlag::AtomicCapable)) == 0u) {
      if (diagnostics) diagnostics->push_back(diag("registry.atomic", n.rid)); return false;
    }
    if (n.lane == BcirLane::A && !is_atomic(n.opcode)) { if (diagnostics) diagnostics->push_back(diag("lane.atomic", "A lane requires atomic opcode")); return false; }
  }
  return true;
}

bool verify_hazards(const BcirGraph& graph, std::vector<Diagnostic>* diagnostics) {
  std::unordered_map<std::string, std::uint64_t> lastWrite;
  for (const auto& n : graph.nodes) {
    const bool isWrite = n.opcode == BcirOpcode::Store || is_atomic(n.opcode);
    const bool isRead = n.opcode == BcirOpcode::Load || is_atomic(n.opcode);
    if (isRead && lastWrite.count(n.rid) && n.hazardMode == BcirHazardMode::Relaxed) {
      if (diagnostics) diagnostics->push_back(diag("hazard.raw", "RAW hazard requires ordering/contract"));
      return false;
    }
    if (isWrite) lastWrite[n.rid] = n.id;
  }
  return true;
}

BcirSchedule build_schedule(const BcirGraph& graph) {
  BcirSchedule s;
  std::vector<const BcirNode*> nodes; for (const auto& n : graph.nodes) nodes.push_back(&n);
  std::sort(nodes.begin(), nodes.end(), [](const BcirNode* a, const BcirNode* b) {
    return std::tie(a->epoch, a->phase, a->lane, a->opcode, a->hazardDomain, a->id) < std::tie(b->epoch, b->phase, b->lane, b->opcode, b->hazardDomain, b->id);
  });
  std::uint64_t packed = UINT64_MAX;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const BcirNode* n = nodes[i];
    s.executionOrder.push_back(n->id);
    const std::uint64_t now = (static_cast<std::uint64_t>(n->epoch) << 32u) | n->phase;
    if (now != packed) {
      s.phaseBegin.push_back(i);
      if (!s.phaseEnd.empty()) s.phaseEnd.back() = i;
      s.phaseEnd.push_back(nodes.size());
      packed = now;
    }
  }
  if (!s.phaseEnd.empty()) s.phaseEnd.back() = nodes.size();
  return s;
namespace bcir {

BcirCoreBuildResult build_core_graph(const BlockNode& block) {
  BcirCoreBuildResult out;
  std::uint32_t epoch = 0;
  std::uint32_t phase = 0;
  for (std::size_t i = 0; i < block.operations.size(); ++i) {
    const auto* op = block.operations[i].get();
    BcirNode node;
    node.id = i;
    node.epoch = epoch;
    node.phase = phase;
    if (op->kind == Operation::Kind::Phase) {
      const auto* p = static_cast<const PhaseOperation*>(op);
      phase = static_cast<std::uint32_t>(p->phase);
      node.phase = phase;
      node.opcode = BcirOpcode::Phase;
      node.opcodeText = "phase";
    } else if (op->kind == Operation::Kind::Barrier) {
      node.opcode = BcirOpcode::Barrier;
      node.opcodeText = "barrier";
    } else if (op->kind == Operation::Kind::LdSt) {
      const auto* l = static_cast<const LdStOperation*>(op);
      node.registry = l->target;
      node.opcode = l->isLoad ? BcirOpcode::Load : BcirOpcode::Store;
      node.opcodeText = l->isLoad ? "ld" : "st";
    } else if (op->kind == Operation::Kind::Binary) {
      const auto* b = static_cast<const BinaryOpOperation*>(op);
      node.opcodeText = b->opcode;
      node.opcode = b->opcode == "add" ? BcirOpcode::Add : BcirOpcode::Unknown;
    } else if (op->kind == Operation::Kind::MapSurface) {
      const auto* m = static_cast<const MapSurfaceOperation*>(op);
      node.registry = m->target;
      switch (m->surfaceKind) {
        case MapSurfaceOperation::SurfaceKind::Load: node.opcode = BcirOpcode::Load; node.opcodeText = "map_load"; break;
        case MapSurfaceOperation::SurfaceKind::Store: node.opcode = BcirOpcode::Store; node.opcodeText = "map_store"; break;
        case MapSurfaceOperation::SurfaceKind::AtomicAdd: node.opcode = BcirOpcode::AtomicAdd; node.opcodeText = "map_atomic_add"; break;
        case MapSurfaceOperation::SurfaceKind::AtomicSub: node.opcode = BcirOpcode::AtomicSub; node.opcodeText = "map_atomic_sub"; break;
        case MapSurfaceOperation::SurfaceKind::AtomicXor: node.opcode = BcirOpcode::AtomicXor; node.opcodeText = "map_atomic_xor"; break;
      }
    }
    if (!out.graph.nodes.empty()) {
      out.graph.edges.push_back({out.graph.nodes.back().id, node.id, BcirEdgeKind::ControlFlow});
      if (out.graph.nodes.back().phase != node.phase) out.graph.edges.push_back({out.graph.nodes.back().id, node.id, BcirEdgeKind::PhaseOrder});
    }
    out.graph.nodes.push_back(node);
  }
  return out;
}

}  // namespace bcir
