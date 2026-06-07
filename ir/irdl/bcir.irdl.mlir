// BCIR dialect — IRDL projection (pure IR, no compilation).
//
// Load with:  mlir-opt --irdl-file=bcir.irdl.mlir <program-in-bcir-dialect.mlir>
//
// This file is the SCAFFOLD for the declarative BCIR dialect definition. It is
// intentionally minimal and valid so `mlir-opt bcir.irdl.mlir` round-trips. The
// full type/op/enum/constraint projection of ir/core/include/bcir/bcir_ir.hpp is the
// next implementation task (see ir/irdl/README.md and docs/BCIR_Repo_Structure.md).
//
// Source of truth being projected (ir/core/include/bcir/bcir_ir.hpp):
//   lanes:      U, UX, T, GGG, A, H              (BcirLane)
//   edge kinds: DataFlow, ControlFlow, HazardOrder, PhaseOrder, CostFlow
//   hazards:    RAW, WAR, WAW                    (BcirHazardKind)
//   contracts:  PromiseNoHazard, RequiresOrdering
//   opcodes:    Load, Store, Add, ..., AtomicAdd, CmpXchg, Barrier, Phase
//
// IRDL CAN encode: type/operand/result typing, attribute presence, closed enum
// membership, scalar range bounds. IRDL CANNOT encode cross-op invariants
// (phase monotonicity, epoch/phase legality, hazard contracts, concurrent
// registry/atomic rules) — those remain in the C++ surface verifier.

irdl.dialect @bcir {
  // --- Types (placeholders; parameters added during projection) -------------
  // TODO(bcir-irdl): add parameters, e.g. @vertex<space, id_bits>.
  irdl.type @vertex
  irdl.type @edge
  irdl.type @graph

  // --- Operations (placeholders; operands/results/attrs added next) ---------
  // TODO(bcir-irdl): project bcir.load/store/bin/lane/phase/barrier + map_* +
  // graph ops, wiring operands/results/attributes with enum constraints.
  irdl.operation @barrier
  irdl.operation @phase
}
