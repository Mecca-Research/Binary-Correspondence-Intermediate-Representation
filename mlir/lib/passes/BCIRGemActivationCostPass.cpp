//===- BCIRGemActivationCostPass.cpp - the -bcir-gem-activation-cost roofline ---*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
// Recomputes the G1 activation ROOFLINE COST for each `bcir.gem.activation` plan op (the
// law-rail port of bcir/kbcir/activation.py::cost_of) and annotates the recomputed
// compute/mem/bottleneck terms on the op. Like -bcir-gem-matmul-cost / -bcir-layout-pivot /
// -bcir-cache-contention this is a cost PRODUCER, not a lowering (cf. -bcir-lower-gem-
// activation, which ERASES the activation into its lane-width stripes): the op is NOT erased --
// the recomputed roofline is annotated as kbcir.* attrs (the cross-pass dual-rail record).
//
// THE ROOFLINE THIS PASS REPRODUCES (mirrors activation.py::cost_of exactly, verified by
// mlir/test/passes/gem_activation_cost.mlir + bcir/tests/test_activation_law_parity.py):
//   An activation is an elementwise tensor op over `count` = product(shape) elements. cost_of is
//   the SAME roofline shape matmul.cost_of uses, specialized to an elementwise op:
//     op_weight = {relu:1, sigmoid:8, tanh:8, gelu:12, softmax:8}[kind] -- the per-element compute
//                 weight (relu's single max == 1; a transcendental ~ a libm call modeled at 8x; gelu
//                 is heavier still at 12; softmax adds the exp), so the planner ranks them honestly.
//     compute = ceilDiv(count*op_weight, vector_width*(2 if fma else 1)) -- the throughput term
//     streams = 3 if softmax else 2  (softmax re-reads the row for its reduce-max + normalize passes;
//                 the elementwise four are 1 read + 1 write per element)
//     mem     = ceilDiv(count*streams, mem_unit*mem_channels)            -- the bandwidth term
//     bottleneck = max(compute, mem)                                     -- the max,+ binding resource
//   (cost_of does NOT depend on the chosen lane `width` -- the width drives only the stripe count in
//   the lowering; the roofline is shape/kind-only, exactly as the oracle computes it.)
//
// DETERMINISM (host-independence): the oracle cost depends on the TargetProfile, so this pass
// PINS the reference constants to TargetProfile.x86_avx512() (named consts below), EXACTLY as
// the sibling cost passes + their *_law_parity.py tests pin x86-avx512 so the recomputed cost
// is CI-host-independent (TargetProfile.for_host() would price the roofline differently per arch).
//
// THE PARITY THE VERIFIER OMITS: GEMActivationOp::verify() checks only well-formedness (a known
// kind, positive shape extents, a known + preserved dtype, the quarantine dtype rule, the softmax
// axis, width in [1, count], the R17 acc_bound, and bottleneck == max(compute, mem)) -- it does
// NOT recompute the ANALYTIC cost from the shape/kind. This pass adds exactly that: it ports
// cost_of, cross-checks the declared compute_cost/mem_cost/bottleneck, and emitError on a mismatch
// (the dual-rail R13 parity gate). INFORMS-ONLY: the recomputed roofline informs the plan's cost
// but is NEVER read by -bcir-verify as a legality verdict -- the activation op is NOT erased and
// never gates legality (the same two-truth quarantine the sibling cost passes keep).
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRPassSupport.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>

using namespace mlir;

namespace bcir {
namespace {

// ceil(a / b) for positive b (mirrors activation.py's (x + d - 1)//d and the BCIRGemMatmulCostPass
// ceilDiv helper).
static int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// The per-element compute weight cost_of charges for `kind` (activation.py's `op_weight` map): relu
// is a single max (1); a transcendental ~ a libm call modeled at 8x; gelu is heavier (a cube + a
// tanh) at 12; softmax adds the exp at 8. Returns -1 for an unknown kind (the op verifier rejects
// such an op before this pass, but the cost port mirrors the oracle's exhaustive table defensively).
static int64_t opWeight(StringRef kind) {
  if (kind == "relu") return 1;
  if (kind == "sigmoid") return 8;
  if (kind == "tanh") return 8;
  if (kind == "gelu") return 12;
  if (kind == "softmax") return 8;
  return -1;
}

struct GemActivationCostPass
    : public PassWrapper<GemActivationCostPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemActivationCostPass)

  StringRef getArgument() const final { return "bcir-gem-activation-cost"; }
  StringRef getDescription() const final {
    return "Recompute the G1 activation roofline (port of activation.py::cost_of) for each "
           "gem.activation op and parity-check the declared compute/mem/bottleneck; annotate "
           "kbcir.* INFORMS-ONLY (never a legality verdict).";
  }

  void runOnOperation() override {
    SmallVector<GEMActivationOp> ops;
    getOperation()->walk([&](GEMActivationOp act) { ops.push_back(act); });
    bool ok = true;
    for (GEMActivationOp act : ops)
      ok &= priceOne(act);
    if (!ok)
      signalPassFailure();
  }

  // Recompute the roofline cost from the op's kind + shape through the pinned x86-avx512 reference
  // constants, cross-check the declared compute/mem/bottleneck, and annotate the recomputed
  // roofline. Returns false on a parity violation. (The op is NOT erased -- this is a cost
  // producer, not a lowering.)
  bool priceOne(GEMActivationOp op) {
    MLIRContext *ctx = op.getContext();
    Builder ab(ctx);

    // The pinned TargetProfile.x86_avx512() reference constants (activation.py::cost_of reads these
    // off the target; pinned here for CI-host-independence, EXACTLY as the sibling cost passes +
    // their *_law_parity.py tests pin x86-avx512): vector_width = max(1,8,16) = 16, fma = true,
    // mem_unit = 1, mem_channels = 4. (cost_of uses none of cacheline/elem_bytes.)
    const int64_t kVectorWidth = 16;
    const int64_t kFma = 2;           // (2 if fma else 1); avx512 has fma -> 2
    const int64_t kMemUnit = 1;
    const int64_t kMemChannels = 4;

    StringRef kind = op.getKind();
    int64_t weight = opWeight(kind);
    if (weight < 0) {
      op.emitError("bcir-gem-activation-cost: unknown activation kind '")
          << kind << "' (expected relu|sigmoid|tanh|gelu|softmax)";
      return false;
    }

    // count = product(shape) -- the element count cost_of prices (rank-agnostic, like ActivationSpec.count).
    ArrayRef<int64_t> shape = op.getShape();
    int64_t count = 1;
    for (int64_t d : shape)
      count *= d;

    // --- recompute the analytic roofline (identical to activation.py::cost_of) ------------------
    // Computed in int64 -- the cost domain the GEMActivationOp attrs themselves are declared in (I64Attr) and
    // that the verifier + every sibling cost pass share. n = max(1, count) (a degenerate empty tensor is
    // floored to one element, exactly as cost_of's `n = max(1, spec.count)`).
    int64_t n = std::max<int64_t>(1, count);
    int64_t thr = std::max<int64_t>(1, kVectorWidth * kFma);     // the throughput divisor (vector_width * fma)
    int64_t compute = ceilDiv(n * weight, thr);                  // count*op_weight / (vector_width * fma)
    int64_t streams = (kind == "softmax") ? 3 : 2;               // softmax re-reads the row (max + exp/sum + normalize)
    int64_t bw = std::max<int64_t>(1, kMemUnit * kMemChannels);  // the bandwidth divisor
    int64_t mem = ceilDiv(n * streams, bw);                      // bytes streamed / bandwidth
    int64_t bottleneck = std::max(compute, mem);                 // max,+ : the binding roofline resource

    // --- cross-check the declared roofline (the dual-rail parity gate) ------------------
    if (compute != static_cast<int64_t>(op.getComputeCost())) {
      op.emitError("bcir-gem-activation-cost: compute_cost ")
          << static_cast<int64_t>(op.getComputeCost()) << " != the recomputed roofline compute "
          << compute << " (ceilDiv(count*op_weight, vector_width*fma))";
      return false;
    }
    if (mem != static_cast<int64_t>(op.getMemCost())) {
      op.emitError("bcir-gem-activation-cost: mem_cost ")
          << static_cast<int64_t>(op.getMemCost()) << " != the recomputed roofline mem " << mem
          << " (ceilDiv(count*streams, mem_unit*mem_channels))";
      return false;
    }
    if (bottleneck != static_cast<int64_t>(op.getBottleneck())) {
      op.emitError("bcir-gem-activation-cost: bottleneck ")
          << static_cast<int64_t>(op.getBottleneck()) << " != max(compute, mem) " << bottleneck;
      return false;
    }

    // --- annotate the recomputed roofline (the cross-pass dual-rail record) -------------
    // INFORMS-ONLY: the recomputed roofline informs the plan's cost but never gates legality
    // (-bcir-verify never reads these).
    op->setAttr("kbcir.compute_cost", ab.getI64IntegerAttr(compute));
    op->setAttr("kbcir.mem_cost", ab.getI64IntegerAttr(mem));
    op->setAttr("kbcir.bottleneck", ab.getI64IntegerAttr(bottleneck));
    op->setAttr("kbcir.informs_only", ab.getBoolAttr(true));         // never a legality verdict
    return true;
  }
};

}  // namespace

std::unique_ptr<Pass> createGemActivationCostPass() {
  return std::make_unique<GemActivationCostPass>();
}

}  // namespace bcir
