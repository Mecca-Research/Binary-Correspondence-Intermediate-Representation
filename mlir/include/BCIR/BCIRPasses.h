//===- BCIRPasses.h - BCIR compiler passes -----------------------*- C++ -*-===//
//
// Phase 6: the MLIR rail as a real compiler.
//   -bcir-verify           semantic laws R1-R16 as a module pass
//   -bcir-promote-lanes    the opt-law (GGG -> UX promotion) as a rewrite
//   -convert-bcir-to-llvm  BCIR compute/barrier -> LLVM dialect (TypeConverter + patterns)
//   -bcir-classify-lanes / -select-realization / -rcsp / -batch / -schedule /
//   -lower-to-llvm         the GEM pipeline + RCSP/Pareto (the optimizer core, C++23)
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_BCIRPASSES_H
#define BCIR_BCIRPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace bcir {

std::unique_ptr<mlir::Pass> createVerifyPass();
std::unique_ptr<mlir::Pass> createPromoteLanesPass();
std::unique_ptr<mlir::Pass> createConvertToLLVMPass();

// The GEM pipeline (LangRef Milestone 4..7): classify -> select -> batch ->
// schedule -> lower. MLIR-native implementations of the bcir/ oracle stages,
// cross-checked against its pinned constants (docs/PARITY.md).
std::unique_ptr<mlir::Pass> createClassifyLanesPass();
std::unique_ptr<mlir::Pass> createSelectRealizationPass();
// -bcir-cost-model: the K_BCIR cost algebra (cost.py) on the MLIR rail -- recompute
// candidate costs from claim + target.capability instead of trusting declared paths.
std::unique_ptr<mlir::Pass> createCostModelPass();
// -bcir-plan: the layered min-plus shortest path (the full realize.optimize in C++).
std::unique_ptr<mlir::Pass> createPlanPass();
// -bcir-overlap: the (max,+) scheduled price M(pi,Theta) (gem/overlap.py).
std::unique_ptr<mlir::Pass> createOverlapPass();
// -bcir-overlap-optimize: the makespan-driven re-selection sweep (overlap.py
// ::optimize_scheduled) -- adopt the per-claim alternative that strictly lowers makespan.
std::unique_ptr<mlir::Pass> createOverlapOptimizePass();
// -bcir-sense: regret-driven telemetry resolution gate (kbcir/sensing.py RegretSensor.sense).
std::unique_ptr<mlir::Pass> createSensePass();
// -bcir-rcsp: constrained selection (budget label-DP) + the Pareto front, the
// deterministic optimizer core ported from bcir/kbcir/rcsp.py.
std::unique_ptr<mlir::Pass> createRcspPass();
// -bcir-rcsp-plan: plan-level constrained selection (accumulated-budget label-DP).
std::unique_ptr<mlir::Pass> createRcspPlanPass();
// -bcir-bundle: detect + joint-reorder multi-claim input-sharing bundles (kbcir.bundle).
std::unique_ptr<mlir::Pass> createBundlePass();
// -bcir-explain: the proof-carrying decision record (proof.explain) as IR annotations --
// per claim the candidates weighed + chosen width/score; per module the total plan score.
std::unique_ptr<mlir::Pass> createExplainPass();
// -bcir-replay: recheck the declared kbcir.explain_* record (proof.replay) -- recompute a fresh
// plan and diff (module total + per-claim chosen/score); annotates kbcir.replay_reproduced.
std::unique_ptr<mlir::Pass> createReplayPass();
// -bcir-compose: compositional cost over the kbcir.func/call/cond region tree
// (compose.plan_composite) -- annotates kbcir.compose_worst / compose_expected per func.
std::unique_ptr<mlir::Pass> createComposePass();
// -bcir-cim / -bcir-dvfs: recompute the CIM/PIM dispatch + DVFS clock decisions (gem.cim /
// gem.dvfs) from the IR, instead of R14/R15 only verifying a declared attr.
std::unique_ptr<mlir::Pass> createCimPass();
std::unique_ptr<mlir::Pass> createDvfsPass();
// -bcir-schedule-eft: duration-aware EFT wave scheduling (gem.schedule.schedule_eft).
std::unique_ptr<mlir::Pass> createScheduleEftPass();
// -bcir-async: async fork/await plan + pipelined cross-phase schedule (gem.async_tokens +
// schedule.execute_tokens) -- later-phase independent claims overlap earlier ones.
std::unique_ptr<mlir::Pass> createAsyncPass();
// -bcir-power-rail: per-slot DVFS over the EFT placed timeline (gem.schedule.schedule_power_rail)
// -- classify + clock each scheduled slot's interval (the join of -bcir-schedule-eft + -bcir-dvfs).
std::unique_ptr<mlir::Pass> createPowerRailPass();
// -bcir-alloc-pool: liveness-based memory pool planning (kbcir.allocator.pool_plan).
std::unique_ptr<mlir::Pass> createAllocPoolPass();
std::unique_ptr<mlir::Pass> createBatchPass();
std::unique_ptr<mlir::Pass> createSchedulePass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
// -bcir-lower-gem-matmul: lower each gem.matmul plan record into its concrete
// tiled realization -- one gem.block descriptor per tile of the tiled iteration
// space, emitted in the plan's loop_order; erases the matmul (a genuine lowering).
std::unique_ptr<mlir::Pass> createLowerGemMatmulPass();
// -bcir-lower-gem-matmul-buffer: lower each gem.matmul_buffer (real SSA memref
// operands A/B/C + tile plan) into a concrete tiled scf.for loop nest computing
// C += A*B (memref.load/store + arith.mulf/addf); erases the op (genuine lowering).
std::unique_ptr<mlir::Pass> createLowerGemMatmulBufferPass();
// -bcir-lower-gem-activation: lower each gem.activation plan record (the K_BCIR-chosen
// lane-width realization of relu/sigmoid/tanh/gelu/softmax, G1) into its concrete
// realization -- one gem.block per lane-width stripe; recomputes the dual-semiring
// roofline cost + the quarantine/R17 verdict (relu exact, the transcendentals route a
// libm edge); erases the activation (the activation analog of lower-gem-matmul).
std::unique_ptr<mlir::Pass> createLowerGemActivationPass();
// -bcir-lower-gem-conv: lower each gem.conv plan record (the K_BCIR-chosen direct|im2col
// realization of a 2-D single-group convolution, G7) into its concrete realization -- the
// EQUIVALENT im2col gemm tiled into one gem.block per tile, in loop_order (a conv IS a
// structured matmul, priced through the matmul roofline -- no bespoke term). Recomputes the
// host-independent roofline parity (bottleneck == max) + the R17 verdict, reproduces
// plan_conv's strategy (direct = the untiled gemm), and erases the conv (the conv analog of
// lower-gem-matmul / lower-gem-activation).
std::unique_ptr<mlir::Pass> createLowerGemConvPass();
// -bcir-lower-gem-attention: lower each gem.attention plan record (the K_BCIR-chosen
// two-matmul realization of single-head scaled-dot-product attention
// softmax(Q@K^T/sqrt(d_k))@V, G7) into its concrete decomposition -- the two gem.matmul
// tile sequences (scores Q@K^T + context A@V) with the softmax gem.activation between
// them. Attention DECOMPOSES into the existing ops, so it is priced through the matmul
// roofline for BOTH matmuls (the SUM of the two priced costs -- no bespoke term).
// Recomputes the host-independent roofline parity (summed compute/mem == the two
// per-gemm costs; bottleneck == max) + the R17 verdict + the softmax quarantine (expf
// via the trusted libm edge; the matmuls are exact), and erases the attention (the
// attention analog of lower-gem-matmul / lower-gem-conv).
std::unique_ptr<mlir::Pass> createLowerGemAttentionPass();
// -bcir-fuse-matmul-activation: the G2 matmul+activation epilogue fusion (port of
// bcir/kbcir/fusion.py::optimize_fused). Fuse a SOLE-CONSUMER gem.matmul -> gem.activation
// pair into a single gem.fused_matmul_activation iff the deforestation-priced fused score
// (the intermediate's mem round-trip elided, x0.75) STRICTLY beats the unfused score (a
// barrier materializes it). Carries the quarantine split through the epilogue (relu exact;
// the transcendentals route a libm edge) and scopes out softmax (a non-fusible row
// reduction); a non-fusible / unprofitable / non-sole-consumer pair is a clean no-op.
std::unique_ptr<mlir::Pass> createFuseMatmulActivationPass();

/// Register all BCIR passes with the global pass registry (for bcir-opt).
void registerBCIRPasses();

/// Register the named pass pipelines (bcir-audit / -optimize / -hydrate /
/// -lower-llvm / -aot) with verifier checkpoints.
void registerBCIRPipelines();

}  // namespace bcir

#endif  // BCIR_BCIRPASSES_H
