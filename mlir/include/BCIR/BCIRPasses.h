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
// -bcir-gem-matmul-cost: recompute the B1 matmul ROOFLINE COST for each gem.matmul
// plan op (port of bcir/kbcir/matmul.py::cost_of) -- compute = ceilDiv(M*N*K,
// vector_width*fma), mem = ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels),
// bottleneck = max(compute, mem) -- pinning the x86-avx512 reference constants for
// CI-host-independence, parity-check the declared compute/mem/bottleneck (the analytic
// parity the op verifier deliberately omits), and annotate the recomputed roofline (a cost
// producer, NOT a lowering: the op is NOT erased). INFORMS-ONLY: the recomputed roofline
// informs the plan's cost but NEVER gates legality (the same two-truth quarantine the
// sibling cost passes -cache-contention / -layout-pivot keep; -bcir-verify never reads these).
std::unique_ptr<mlir::Pass> createGemMatmulCostPass();
// -bcir-gem-activation-cost: recompute the G1 activation ROOFLINE COST for each gem.activation
// plan op (port of bcir/kbcir/activation.py::cost_of) -- compute = ceilDiv(count*op_weight,
// vector_width*fma), mem = ceilDiv(count*streams, mem_unit*mem_channels), bottleneck =
// max(compute, mem) (op_weight = relu 1 / sigmoid|tanh|softmax 8 / gelu 12; streams = 3 softmax
// else 2) -- pinning the x86-avx512 reference constants for CI-host-independence, parity-check
// the declared compute/mem/bottleneck (the analytic parity the op verifier deliberately omits),
// and annotate the recomputed roofline (a cost producer, NOT a lowering: the op is NOT erased).
// INFORMS-ONLY: the recomputed roofline informs the plan's cost but NEVER gates legality (the same
// two-truth quarantine the sibling cost passes -gem-matmul-cost / -cache-contention / -layout-pivot
// keep; -bcir-verify never reads these).
std::unique_ptr<mlir::Pass> createGemActivationCostPass();
// -bcir-gem-conv-cost: recompute the G7 conv ROOFLINE COST for each gem.conv plan op
// (port of bcir/kbcir/conv.py::cost_of). A 2-D single-group conv is a STRUCTURED MATMUL
// priced through the matmul roofline (NO bespoke term): cost_of calls matmul.cost_of on the
// equivalent im2col gemm dims (gemm_m = out_h*out_w, gemm_n = out_c, gemm_k = in_c*kh*kw) at
// the resolved tile -- compute = ceilDiv(M*N*K, vector_width*fma), mem = ceilDiv(a_reads +
// b_reads + c_traffic, mem_unit*mem_channels), bottleneck = max(compute, mem) (the im2col
// strategy prices at the chosen inner matmul tile; direct is the UNTILED whole gemm). Pinning
// the x86-avx512 reference constants for CI-host-independence, parity-check the declared
// compute/mem/bottleneck (the analytic parity the op verifier deliberately omits), and annotate
// the recomputed roofline (a cost producer, NOT a lowering: the op is NOT erased). INFORMS-ONLY:
// the recomputed roofline informs the plan's cost but NEVER gates legality (the same two-truth
// quarantine the sibling cost passes -gem-matmul-cost / -gem-activation-cost / -cache-contention /
// -layout-pivot keep; -bcir-verify never reads these).
std::unique_ptr<mlir::Pass> createGemConvCostPass();
// -bcir-gem-attention-cost: recompute the G7 attention ROOFLINE COST for each gem.attention plan op
// (port of bcir/kbcir/attention.py::cost_of). Single-head scaled-dot-product attention DECOMPOSES
// into two gem.matmuls (a softmax + scale ride between them), so it is priced through the matmul
// roofline for BOTH matmuls (NO bespoke term): cost_of calls matmul.cost_of on the scores Q@K^T gemm
// (seq, seq, d_k) and the context A@V gemm (seq, d_k, seq) at their chosen tiles and SUMS them --
// compute_cost = scores_compute + context_compute, mem_cost = scores_mem + context_mem (each per-gemm
// term = ceilDiv(M*N*K, vector_width*fma) / ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels)),
// bottleneck = max(compute, mem). Pinning the x86-avx512 reference constants for CI-host-independence,
// parity-check the declared per-gemm + summed compute/mem/bottleneck (the analytic parity the op verifier
// deliberately omits), and annotate the recomputed roofline (a cost producer, NOT a lowering: the op is
// NOT erased). INFORMS-ONLY: the recomputed roofline informs the plan's cost but NEVER gates legality (the
// same two-truth quarantine the sibling cost passes -gem-matmul-cost / -gem-activation-cost / -gem-conv-cost
// / -cache-contention / -layout-pivot keep; -bcir-verify never reads these).
std::unique_ptr<mlir::Pass> createGemAttentionCostPass();
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
// -bcir-cache-contention: the G4 cache-line/memory-bank-conflict CONTENTION cost SIGNAL
// (port of bcir/kbcir/cache_predict.py::CachePredictor.contention_q8). For each
// gem.contention carrier op it RECOMPUTES the frozen analytic signal from the access shape
// (stride/elem_bytes/count) + frozen cache/bank params -- line_waste_q8 = (min(stride,
// cacheline/elem)-1)*256, bank_conflict_q8 = (gcd(stride, banks)-1)*256, contention_q8 = the
// sum, contention_cost = (contention_q8*count)>>8 -- and annotates them (a cost producer, NOT
// a lowering: the op is not erased). INFORMS-ONLY: the signal rides the CONTENTION cost axis
// and NEVER gates legality (the two-truth quarantine -- -bcir-verify reads only R8/R9).
std::unique_ptr<mlir::Pass> createCacheContentionPass();
// -bcir-layout-pivot: the G3 SoA<->AoS layout-pivot PRICING DECISION (port of
// bcir/kbcir/layout.py::plan_layout + LayoutCertificate). For each gem.layout_pivot carrier op
// it prices SoA vs AoS through the SAME stride-penalty memory terms realize.candidates_for
// prices every claim by (a single-field sweep is UNIT under SoA / STRIDED(F) under AoS; a
// whole-record access the mirror), picks the MIN-cost layout (ties keep the default SoA), and
// annotates the chosen layout + the SoA/AoS cost + the gain (a LayoutCertificate-equivalent; a
// cost producer, NOT a lowering). The hard "layout changes addresses, never values" invariant
// is the oracle's; the law records the priced choice. INFORMS-ONLY (never a legality verdict).
std::unique_ptr<mlir::Pass> createLayoutPivotPass();

/// Register all BCIR passes with the global pass registry (for bcir-opt).
void registerBCIRPasses();

/// Register the named pass pipelines (bcir-audit / -optimize / -hydrate /
/// -lower-llvm / -aot) with verifier checkpoints.
void registerBCIRPipelines();

}  // namespace bcir

#endif  // BCIR_BCIRPASSES_H
