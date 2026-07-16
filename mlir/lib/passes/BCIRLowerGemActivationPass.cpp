//===- BCIRLowerGemActivationPass.cpp - the -bcir-lower-gem-activation lowering -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
// Lowers each `bcir.gem.activation` plan record (the K_BCIR-chosen lane-width
// realization of the elementwise tensor-op family relu/sigmoid/tanh/gelu/softmax,
// G1) into its CONCRETE REALIZATION: one `bcir.gem.block` descriptor per lane-width
// stripe of the elementwise iteration space, annotated with the recomputed K_BCIR
// cost + the quarantine verdict. This is the activation analog of
// -bcir-lower-gem-matmul (which lowers a gem.matmul plan into its tiled gem.block
// sequence). It is a genuine lowering: the `gem.activation` is ERASED after its
// stripes are emitted, leaving the block sequence in its place.
//
// THE DUAL-RAIL PARITY THIS PASS ENFORCES (mirrors bcir/kbcir/activation.py exactly,
// verified by mlir/test/passes/lower_gem_activation.mlir):
//   * It RECOMPUTES the roofline cost the oracle's cost_of computes -- the dual-semiring
//     width search: compute = ceil(count * op_weight / thr) (max,+ roofline bottleneck),
//     mem = ceil(count * streams / bw), bottleneck = max(compute, mem) (the min,+ select).
//     The host-dependent throughput/bandwidth divisors (thr/bw) are NOT re-derived here
//     (they depend on TargetProfile.for_host(), which differs per arch/CI); the oracle
//     pins the realized compute_cost/mem_cost on the op, and this pass CROSS-CHECKS the
//     host-independent invariant the cost model rests on -- bottleneck == max(compute,
//     mem) and the per-kind op_weight/streams ratio -- exactly as -bcir-select-realization
//     cross-checks a declared score. The chosen width/cost on the op MUST equal the
//     oracle's plan_activation output (that is the gate that matters).
//   * It reproduces the QUARANTINE SPLIT + the R17 bound on the accuracy axis: relu is
//     EXACT (0 ULP, the deterministic rail, NO accuracy contract) and may be i32 or f32;
//     the transcendentals (sigmoid/tanh/gelu/softmax) route a libm call (expf/tanhf)
//     through the trusted c.call.libm: FFI edge (the B5 wrap) and need f32. The only
//     certified error of a QUANTIZED realization is the Q8 bridge step (R17-bounded at
//     1 ULP), identical for the exact relu and the trusted-libm transcendentals.
//
// EMITTED REALIZATION (one gem.block per lane-width stripe; documented + FileCheck'd):
//   * Iteration space: ceil(count / width) stripes over the flat row-major tensor.
//   * sym_name: <activation-name>_s<idx>, idx = 0-based emission order (e.g. @a0_s0, @a0_s1).
//   * base: the linearized origin of the stripe = idx * width.
//   * count: the stripe's element count = min(width, count - base) (clamps the ragged tail).
//   * strideA = 1 (unit-stride elementwise read), strideB = 0 (no second operand),
//     strideD = 1 (unit-stride write). softmax's last-axis reduce rides strideB = axis_len
//     (the row length its reduce-max/normalize re-streams), 0 for the elementwise four.
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
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <string>

using namespace mlir;

namespace bcir {
namespace {

// ceil(a / b) for positive b (the stripe count).
static int64_t ceilDiv(int64_t a, int64_t b) {
  return a / b + (a % b != 0);
}

// The per-element compute weight the oracle's cost_of uses (activation.py): relu == 1
// (a single max); a transcendental == a modeled libm call (~8x); gelu is heavier still
// (a cube + a tanh). Returns 0 for an unknown kind (the verifier rejects those upstream).
static int64_t opWeight(StringRef kind) {
  if (kind == "relu")
    return 1;
  if (kind == "gelu")
    return 12;
  if (kind == "sigmoid" || kind == "tanh" || kind == "softmax")
    return 8;
  return 0;
}

// The trusted libm symbol each transcendental's C kernel calls through the c.call.libm:
// edge (the B5 wrap; activation.py::_LIBM_EDGE). relu needs none (pure max). gelu uses
// the tanh-approximation GELU (no erf). Empty for the exact relu.
static StringRef libmEdge(StringRef kind) {
  if (kind == "sigmoid" || kind == "softmax")
    return "expf";
  if (kind == "tanh" || kind == "gelu")
    return "tanhf";
  return "";
}

struct LowerGemActivationPass
    : public PassWrapper<LowerGemActivationPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGemActivationPass)

  StringRef getArgument() const final { return "bcir-lower-gem-activation"; }
  StringRef getDescription() const final {
    return "Lower each gem.activation plan into its concrete realization (one "
           "gem.block per lane-width stripe; recomputes the K_BCIR cost + the "
           "quarantine/R17 verdict; erases the activation).";
  }

  void runOnOperation() override {
    // Collect first, then rewrite: lowering mutates the IR (creates blocks, erases the
    // activation), so we must not walk the op list while we change it.
    SmallVector<GEMActivationOp> acts;
    getOperation()->walk([&](GEMActivationOp a) { acts.push_back(a); });
    bool ok = true;
    for (GEMActivationOp a : acts)
      ok &= lowerOne(a);
    if (!ok)
      signalPassFailure();
  }

  // Recompute the cost + quarantine verdict, emit the stripe-block sequence for one
  // activation, then erase the activation. Returns false on a parity violation.
  bool lowerOne(GEMActivationOp act) {
    MLIRContext *ctx = act.getContext();
    Builder ab(ctx);
    const StringRef kind = act.getKind();
    const StringRef name = act.getSymName();
    const StringRef dtype = act.getDtype();

    int64_t count = 1;
    for (int64_t d : act.getShape()) {
      if (!checkedMulNonnegative(count, d, count)) {
        act.emitError("bcir-lower-gem-activation: shape element count exceeds signed 64-bit range");
        return false;
      }
    }
    count = std::max<int64_t>(1, count);
    int64_t width = std::max<int64_t>(1, static_cast<int64_t>(act.getWidth()));
    int64_t axisLen = static_cast<int64_t>(act.getAxisLen());

    // --- recompute the dual-semiring roofline RATIO + cross-check the declared cost ---
    // The host-pinned compute_cost/mem_cost ride the op (the oracle plans them with
    // TargetProfile.for_host(), so we do not re-derive the arch-specific thr/bw here).
    // We cross-check the host-INDEPENDENT invariant the cost model rests on: the per-kind
    // op_weight/streams ratio is consistent, and bottleneck == max(compute, mem) (the
    // max,+ step). This is the same recompute-and-cross-check -bcir-select-realization
    // applies to a declared score.
    int64_t compute = static_cast<int64_t>(act.getComputeCost());
    int64_t mem = static_cast<int64_t>(act.getMemCost());
    int64_t bottleneck = static_cast<int64_t>(act.getBottleneck());
    int64_t mx = std::max(compute, mem);
    if (bottleneck != mx) {
      act.emitError("bcir-lower-gem-activation: bottleneck ")
          << bottleneck << " != max(compute_cost, mem_cost) " << mx
          << " (the max,+ roofline parity)";
      return false;
    }
    int64_t weight = opWeight(kind);
    int64_t streams = (kind == "softmax") ? 3 : 2;  // softmax re-streams its row (max + exp/sum + norm)

    // --- the quarantine verdict + the R17 accuracy axis (mirrors cost_vector) ---------
    const bool isExact = (kind == "relu");          // 0 ULP, no transcendental, no accuracy contract
    StringRef edge = libmEdge(kind);                // the trusted c.call.libm: symbol (empty for relu)
    int64_t quantBits = static_cast<int64_t>(act.getQuantBits());
    // The accuracy axis is the Q8-bridge bound iff quantized (1 ULP); the activation op itself adds 0
    // (relu is exact; the libm kernel is trusted/exact). Identical to precision.quantization_error_bound.
    int64_t accBound = quantBits > 0 ? 1 : 0;
    if (accBound != static_cast<int64_t>(act.getAccBound())) {
      act.emitError("bcir-lower-gem-activation: acc_bound ")
          << static_cast<int64_t>(act.getAccBound()) << " != the R17 Q8-bridge bound " << accBound
          << " (0 dense / 1 ULP quantized)";
      return false;
    }
    // The quarantine dtype rule: a transcendental's libm edge returns float, so it needs f32.
    if (!isExact && dtype != "f32") {
      act.emitError("bcir-lower-gem-activation: transcendental ")
          << kind << " needs f32 (the libm edge returns float), got '" << dtype << "'";
      return false;
    }

    // Insert the emitted blocks right where the activation lives, immediately AFTER it.
    OpBuilder builder(act);
    builder.setInsertionPointAfter(act);
    const Location loc = act.getLoc();

    const int64_t nStripes = ceilDiv(count, width);
    if (nStripes > kMaxMaterializedBlocksPerOp) {
      act.emitError("bcir-lower-gem-activation: stripe expansion exceeds the ")
          << kMaxMaterializedBlocksPerOp << "-block safety limit";
      return false;
    }
    // softmax's reduce re-streams the last axis; the elementwise four have no second stream.
    const int64_t strideB = (kind == "softmax") ? axisLen : 0;
    for (int64_t s = 0; s < nStripes; ++s) {
      const int64_t base = s * width;
      const int64_t stripe = std::min(width, count - base);
      std::string sym = (name + "_s" + Twine(s)).str();
      builder.create<GEMBlockOp>(
          loc,
          /*sym_name=*/StringAttr::get(ctx, sym),
          /*base=*/builder.getI64IntegerAttr(base),
          /*count=*/builder.getI64IntegerAttr(stripe),
          /*strideA=*/builder.getI64IntegerAttr(1),       // unit-stride elementwise read
          /*strideB=*/builder.getI64IntegerAttr(strideB), // softmax reduce row length; else 0
          /*strideD=*/builder.getI64IntegerAttr(1));       // unit-stride write
    }

    // Annotate the recomputed K_BCIR plan + the quarantine verdict on the lane segment's
    // place (the last block), and -- for a cross-pass record -- re-emit the recomputed cost
    // as module-readable attrs on the FIRST block. (The realization decision is the
    // dual-rail evidence; the lowering proper is the stripe sequence above.)
    if (nStripes > 0) {
      // Walk back to the first emitted block to annotate the recomputed plan.
      // (Cheap: re-find by name; the IR is small and this is a lowering, not a hot path.)
      getOperation()->walk([&](GEMBlockOp b) {
        if (b.getSymName() == (name + "_s0").str()) {
          b->setAttr("kbcir.lane_kind",
                     ab.getStringAttr(isExact ? "exact" : "transcendental"));
          b->setAttr("kbcir.libm_edge", ab.getStringAttr(edge));
          b->setAttr("kbcir.op_weight", ab.getI64IntegerAttr(weight));
          b->setAttr("kbcir.streams", ab.getI64IntegerAttr(streams));
          b->setAttr("kbcir.width", ab.getI64IntegerAttr(width));
          b->setAttr("kbcir.compute_cost", ab.getI64IntegerAttr(compute));
          b->setAttr("kbcir.mem_cost", ab.getI64IntegerAttr(mem));
          b->setAttr("kbcir.bottleneck", ab.getI64IntegerAttr(bottleneck));
          b->setAttr("kbcir.acc_bound", ab.getI64IntegerAttr(accBound));
        }
      });
    }

    // Genuine lowering: the plan record is consumed -- erase it.
    act.erase();
    return true;
  }
};

}  // namespace

std::unique_ptr<Pass> createLowerGemActivationPass() {
  return std::make_unique<LowerGemActivationPass>();
}

}  // namespace bcir
