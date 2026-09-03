//===- BCIRFuseMatmulActivationPass.cpp - the -bcir-fuse-matmul-activation fusion -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
// The G2 matmul+activation epilogue fusion on the MLIR LAW RAIL -- the port of the
// ORACLE bcir/kbcir/fusion.py (optimize_fused + FusionCertificate). A `bcir.gem.matmul`
// whose output feeds a SOLE-CONSUMER `bcir.gem.activation` is fused into a single inline
// epilogue: the activation is applied as each matmul output element is produced, so the
// intermediate C-buffer never round-trips through memory (deforestation). On the law rail
// the producer->consumer epilogue is expressed as ADJACENCY: a gem.matmul whose
// IMMEDIATELY following op is a gem.activation is the epilogue pair (the matmul produces,
// the next op consumes). Sole-consumer legality is the immediate adjacency itself -- if
// any other op read the un-activated intermediate it would sit BETWEEN the matmul and the
// activation, breaking the adjacency (mirrors fusion.find_matmul_activation_epilogues'
// single-reader rule). This is the structural law-rail analog of the oracle's RID-based
// readers[] check.
//
// THE LOAD-BEARING DESIGN CONSTRAINT -- fusion is the optimizer's PRICED CHOICE, not a
// hardcoded rewrite (exactly as fusion.optimize_fused). We do NOT invent a bespoke
// "matmul+activation discount": we reuse the EXISTING deforestation discount
// (realize._DEFOREST_FACTOR == x0.75 memory, 192/256). The fusion is adopted iff the
// FUSED score (the intermediate deforested -- the activation's mem traffic discounted
// x0.75) is STRICTLY LESS than the UNFUSED score (a phase barrier materializes the
// intermediate -- the activation pays full mem). Each declared op carries its own realized
// roofline (the per-op bottleneck = max(compute, mem) the gem.matmul / gem.activation
// verifiers already pin); the pass recomputes the two plan scores from those and adopts
// fusion when it wins, EXACTLY as optimize_fused scores the two module shapes through
// `optimize` and adopts iff fused_score < unfused_score. (We recompute the host-INDEPENDENT
// fusion arithmetic on the declared per-op bottlenecks, the same recompute-and-cross-check
// discipline -bcir-lower-gem-activation applies to a declared cost.)
//
//   unfused_score = matmul.bottleneck + activation.bottleneck                 (two passes)
//   fused_mem     = round-down(activation.mem_cost * 192 / 256)               (deforestation x0.75)
//   fused_act_bn  = max(activation.compute_cost, fused_mem)                   (the max,+ roofline)
//   fused_score   = matmul.bottleneck + fused_act_bn                          (one inline pass)
//   ADOPT iff fused_score < unfused_score (a strict win; else a clean no-op)
//
// THE QUARANTINE SPLIT carries THROUGH the fused epilogue, exactly as the standalone
// gem.activation lowering: relu is EXACT (inline max, 0 ULP, no libm, i32 or f32); the
// transcendentals (sigmoid/tanh/gelu) route a libm call (expf/tanhf) through the trusted
// c.call.libm: edge and need f32. softmax is SCOPED OUT (a last-axis row reduction needs
// the whole row before any output element is final -- it can NOT be an inline epilogue):
// the pass SKIPS a matmul->softmax pair (no fusion), mirroring fusion.is_fusible_activation
// + the gem.fused_matmul_activation verifier's rejection of softmax.
//
// On an adopted fusion the pass REWRITES the (matmul, activation) pair into a single
// `bcir.gem.fused_matmul_activation` op carrying the matmul tile + the activation kind/
// quarantine + the proof-carrying decision (unfused/fused score + gain), then ERASES both
// the matmul and the activation (a genuine fusion). A non-fusible / unprofitable / non-sole-
// consumer pair is left untouched (a clean no-op, mirroring optimize_fused).
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRPassSupport.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>

using namespace mlir;

namespace bcir {
namespace {

// The deforestation discount (realize._DEFOREST_FACTOR): x0.75 on the MEMORY axis (192/256),
// identity elsewhere. Applied to the consumer activation's mem traffic when the intermediate
// is deforested (the inline epilogue elides its write+read round-trip). Q8 fixed-point, so
// the discount is integer floor(mem * 192 / 256) -- the same arithmetic Candidate.couple uses.
static int64_t deforestMem(int64_t mem) {
  return saturatingMulNonnegative(mem, 192) / 256;
}

// True iff `kind` is an ELEMENTWISE activation that can be an inline epilogue. softmax (a
// last-axis row reduction) is NOT fusible (mirrors fusion.is_fusible_activation /
// FUSIBLE_ACTIVATIONS == {relu, sigmoid, tanh, gelu}).
static bool isFusibleActivation(StringRef kind) {
  return kind == "relu" || kind == "sigmoid" || kind == "tanh" || kind == "gelu";
}

struct FuseMatmulActivationPass
    : public PassWrapper<FuseMatmulActivationPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMatmulActivationPass)

  StringRef getArgument() const final { return "bcir-fuse-matmul-activation"; }
  StringRef getDescription() const final {
    return "Fuse a sole-consumer gem.matmul -> gem.activation epilogue into a "
           "gem.fused_matmul_activation, iff the deforestation-priced fused score "
           "strictly beats the unfused score (the G2 port of fusion.optimize_fused; "
           "skips softmax + non-sole-consumer pairs; emits a FusionCertificate record).";
  }

  void runOnOperation() override {
    // Collect the adopted epilogues first, then rewrite: fusion mutates the IR (creates the
    // fused op, erases the pair), so we must not be walking the op list as we change it.
    SmallVector<std::pair<GEMMatmulOp, GEMActivationOp>> pairs;
    getOperation()->walk([&](GEMMatmulOp mm) {
      // The producer->consumer epilogue on the law rail is ADJACENCY: the matmul's
      // IMMEDIATELY following op must be a gem.activation (the sole consumer -- a third op
      // reading the un-activated intermediate would sit between them). Anything else after
      // the matmul (or nothing) -> no legal epilogue.
      Operation *next = mm->getNextNode();
      auto act = dyn_cast_or_null<GEMActivationOp>(next);
      if (act)
        pairs.emplace_back(mm, act);
    });
    for (auto &[mm, act] : pairs)
      tryFuse(mm, act);
  }

  // Score the epilogue fused vs unfused (the deforestation-priced decision) and, iff fusion
  // strictly wins, rewrite the pair into a gem.fused_matmul_activation. A non-fusible
  // (softmax) or unprofitable pair is left untouched -- a clean no-op, mirroring optimize_fused.
  void tryFuse(GEMMatmulOp mm, GEMActivationOp act) {
    const StringRef kind = act.getKind();
    // softmax + any unknown kind: scoped out of epilogue fusion (a row reduction is not an
    // inline epilogue). Skip -- a clean no-op (mirrors fusion.is_fusible_activation).
    if (!isFusibleActivation(kind))
      return;

    // The deforestation-priced fused vs unfused score, from the declared per-op rooflines.
    const int64_t mmBn = static_cast<int64_t>(mm.getBottleneck());
    const int64_t actBn = static_cast<int64_t>(act.getBottleneck());
    const int64_t actCompute = static_cast<int64_t>(act.getComputeCost());
    const int64_t actMem = static_cast<int64_t>(act.getMemCost());

    // UNFUSED: the two passes run separately (a barrier materializes the intermediate; the
    // activation pays full mem). FUSED: the activation's mem traffic is deforested x0.75, and
    // its roofline bottleneck is recomputed as max(compute, deforested mem) -- the max,+ step.
    int64_t unfusedScore;
    if (!checkedAddNonnegative(mmBn, actBn, unfusedScore)) {
      act.emitError("bcir-fuse-matmul-activation: unfused score exceeds signed 64-bit range");
      signalPassFailure();
      return;
    }
    const int64_t fusedActBn = std::max(actCompute, deforestMem(actMem));
    int64_t fusedScore;
    if (!checkedAddNonnegative(mmBn, fusedActBn, fusedScore)) {
      act.emitError("bcir-fuse-matmul-activation: fused score exceeds signed 64-bit range");
      signalPassFailure();
      return;
    }
    const int64_t gain = unfusedScore - fusedScore;

    // Adopt fusion iff it STRICTLY lowers the score (a no-op when it does not help -- e.g. a
    // degenerate shape where the discount rounds away). Mirrors optimize_fused's strict <.
    if (gain <= 0)
      return;

    MLIRContext *ctx = mm.getContext();
    OpBuilder builder(mm);
    builder.setInsertionPoint(mm);
    Builder ab(ctx);

    // The fused op: the matmul tile half (copied verbatim off the producer) + the activation
    // epilogue half (kind + dtype) + the proof-carrying fusion decision (the FusionCertificate
    // analog: unfused/fused score + gain). sym_name = <matmul>_<activation>_fused.
    std::string sym = (mm.getSymName() + "_" + act.getSymName() + "_fused").str();
    GEMFusedMatmulActivationOp::create(builder,
        mm.getLoc(),
        /*sym_name=*/StringAttr::get(ctx, sym),
        /*m=*/ab.getI64IntegerAttr(static_cast<int64_t>(mm.getM())),
        /*n=*/ab.getI64IntegerAttr(static_cast<int64_t>(mm.getN())),
        /*k=*/ab.getI64IntegerAttr(static_cast<int64_t>(mm.getK())),
        /*tile_m=*/ab.getI64IntegerAttr(static_cast<int64_t>(mm.getTileM())),
        /*tile_n=*/ab.getI64IntegerAttr(static_cast<int64_t>(mm.getTileN())),
        /*tile_k=*/ab.getI64IntegerAttr(static_cast<int64_t>(mm.getTileK())),
        /*loop_order=*/ab.getStringAttr(mm.getLoopOrder()),
        /*kind=*/ab.getStringAttr(kind),
        /*dtype=*/ab.getStringAttr(act.getDtype()),
        /*unfused_score=*/ab.getI64IntegerAttr(unfusedScore),
        /*fused_score=*/ab.getI64IntegerAttr(fusedScore),
        /*gain=*/ab.getI64IntegerAttr(gain));

    // Genuine fusion: the producer + the consumer are consumed -- erase both, leaving the
    // single fused epilogue in their place.
    act.erase();
    mm.erase();
  }
};

}  // namespace

std::unique_ptr<Pass> createFuseMatmulActivationPass() {
  return std::make_unique<FuseMatmulActivationPass>();
}

}  // namespace bcir
