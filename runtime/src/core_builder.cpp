#include "bcir/core_builder.hpp"

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
