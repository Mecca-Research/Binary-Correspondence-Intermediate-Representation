#include "bcir/core_builder.hpp"

#include <cstdint>
#include <unordered_map>

namespace bcir {
namespace {

BcirLane lane_from_int(int lane) {
  switch (lane) {
    case 0: return BcirLane::U;
    case 1: return BcirLane::UX;
    case 2: return BcirLane::T;
    case 3: return BcirLane::GGG;
    case 4: return BcirLane::A;
    case 5: return BcirLane::H;
    default: return BcirLane::U;
  }
}

BcirOpcode opcode_from_binary(const std::string& opcode) {
  if (opcode == "add") return BcirOpcode::Add;
  if (opcode == "sub") return BcirOpcode::Sub;
  if (opcode == "mul") return BcirOpcode::Mul;
  if (opcode == "and") return BcirOpcode::And;
  if (opcode == "or") return BcirOpcode::Or;
  if (opcode == "xor") return BcirOpcode::Xor;
  return BcirOpcode::Unknown;
}

void maybe_add_dataflow_edge(BcirGraph* graph, std::uint64_t src, std::uint64_t dst) {
  if (src == dst) return;
  graph->edges.push_back({src, dst, BcirEdgeKind::DataFlow});
}

}  // namespace

BcirCoreBuildResult build_core_graph(const BlockNode& block) {
  BcirCoreBuildResult out;
  std::uint32_t epoch = 0;
  std::uint32_t phase = 0;
  std::unordered_map<std::string, std::uint64_t> last_def_by_symbol;

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
      node.lane = lane_from_int(l->lane);
      node.opcode = l->isLoad ? BcirOpcode::Load : BcirOpcode::Store;
      node.opcodeText = l->isLoad ? "ld" : "st";
      if (!l->isLoad) {
        auto it = last_def_by_symbol.find(l->rid);
        if (it != last_def_by_symbol.end()) {
          maybe_add_dataflow_edge(&out.graph, it->second, node.id);
          node.operands.push_back(it->second);
        }
      }
      last_def_by_symbol[l->rid] = node.id;
    } else if (op->kind == Operation::Kind::Binary) {
      const auto* b = static_cast<const BinaryOpOperation*>(op);
      node.opcodeText = b->opcode;
      node.opcode = opcode_from_binary(b->opcode);
      if (auto it = last_def_by_symbol.find(b->lhs); it != last_def_by_symbol.end()) {
        maybe_add_dataflow_edge(&out.graph, it->second, node.id);
        node.operands.push_back(it->second);
      }
      if (auto it = last_def_by_symbol.find(b->rhs); it != last_def_by_symbol.end()) {
        maybe_add_dataflow_edge(&out.graph, it->second, node.id);
        node.operands.push_back(it->second);
      }
      last_def_by_symbol[b->dst] = node.id;
    } else if (op->kind == Operation::Kind::MapSurface) {
      const auto* m = static_cast<const MapSurfaceOperation*>(op);
      node.registry = m->target;
      node.lane = lane_from_int(m->lane);
      if (auto it = last_def_by_symbol.find(m->rid); it != last_def_by_symbol.end()) {
        maybe_add_dataflow_edge(&out.graph, it->second, node.id);
        node.operands.push_back(it->second);
      }
      switch (m->surfaceKind) {
        case MapSurfaceOperation::SurfaceKind::Load: node.opcode = BcirOpcode::Load; node.opcodeText = "map_load"; break;
        case MapSurfaceOperation::SurfaceKind::Store: node.opcode = BcirOpcode::Store; node.opcodeText = "map_store"; break;
        case MapSurfaceOperation::SurfaceKind::AtomicAdd: node.opcode = BcirOpcode::AtomicAdd; node.opcodeText = "map_atomic_add"; break;
        case MapSurfaceOperation::SurfaceKind::AtomicSub: node.opcode = BcirOpcode::AtomicSub; node.opcodeText = "map_atomic_sub"; break;
        case MapSurfaceOperation::SurfaceKind::AtomicXor: node.opcode = BcirOpcode::AtomicXor; node.opcodeText = "map_atomic_xor"; break;
      }
      last_def_by_symbol[m->rid] = node.id;
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
