//===- BCIRGemConvCostPass.cpp - the -bcir-gem-conv-cost roofline ---------*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
// Recomputes the G7 conv ROOFLINE COST for each `bcir.gem.conv` plan op (the
// law-rail port of bcir/kbcir/conv.py::cost_of) and annotates the recomputed
// compute/mem/bottleneck terms on the op. Like -bcir-gem-matmul-cost / -bcir-gem-
// activation-cost / -bcir-layout-pivot / -bcir-cache-contention this is a cost
// PRODUCER, not a lowering (cf. -bcir-lower-gem-conv, which ERASES the conv into its
// im2col-gemm tiles): the op is NOT erased -- the recomputed roofline is annotated as
// kbcir.* attrs (the cross-pass dual-rail record).
//
// THE ROOFLINE THIS PASS REPRODUCES (mirrors conv.py::cost_of exactly, verified by
// mlir/test/passes/gem_conv_cost.mlir + bcir/tests/test_conv_law_parity.py):
//   A 2-D single-group conv is a STRUCTURED MATMUL: the im2col (patch-gather) reshaping
//   turns the direct conv into the gemm C[out_pix x C_out] = patches[out_pix x (C_in*Kh*Kw)]
//   @ W[(C_in*Kh*Kw) x C_out], so the conv is PRICED THROUGH THE EXISTING matmul roofline
//   (NO bespoke conv term). conv.py::cost_of calls matmul.cost_of on the equivalent im2col
//   gemm dims (M = out_h*out_w = the output pixels, N = out_c, K = in_c*kh*kw = the taps):
//     compute = ceilDiv(M*N*K, vector_width*(2 if fma else 1))   -- the FLOP-throughput term
//     a_reads = M*K*ceil(N/tile_n)   (the patch matrix re-read once per N-tile -- the im2col
//               materialization traffic IS the gemm's A reads, what reading the patches costs)
//     b_reads = K*N*ceil(M/tile_m)   (the weight matrix re-read once per M-tile)
//     c_traffic = M*N*(ceil(K/tile_k) + 1)
//     mem     = ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels) -- the bandwidth term
//     bottleneck = max(compute, mem)                              -- the max,+ binding resource
//   The STRATEGY only changes the tile the gemm runs in: `im2col` prices at the chosen inner
//   matmul tile (the blocked-gemm reuse); `direct` is the UNTILED gemm (tile = whole M,N,K) --
//   it can not block the reduction the way the im2col gemm tile does, so it streams the input
//   window per output pixel. The op carries the already-resolved tile (the verifier pins
//   direct -> tile == whole gemm), so this pass prices matmul.cost_of at (gemm dims, declared
//   tile) -- IDENTICALLY to conv.cost_of for either strategy.
//
// DETERMINISM (host-independence): the oracle cost depends on the TargetProfile, so this pass
// PINS the reference constants to TargetProfile.x86_avx512() (named consts below), EXACTLY as
// the sibling cost passes + their *_law_parity.py tests pin x86-avx512 so the recomputed cost
// is CI-host-independent (TargetProfile.for_host() would price the roofline differently per arch).
//
// THE PARITY THE VERIFIER OMITS: GEMConvOp::verify() checks only well-formedness (positive
// extents, the kernel fits the padded input, a known + preserved dtype, the DERIVED out_h/out_w
// + im2col gemm dims internally consistent, the strategy is direct|im2col, each inner tile in
// [1, gemm dim] with a known loop order, direct -> the untiled whole gemm, bottleneck ==
// max(compute, mem), and the R17 acc_bound) -- it does NOT recompute the ANALYTIC cost from the
// gemm dims/tile. This pass adds exactly that: it ports cost_of (matmul.cost_of on the im2col
// dims), cross-checks the declared compute_cost/mem_cost/bottleneck, and emitError on a mismatch
// (the dual-rail R13 parity gate). INFORMS-ONLY: the recomputed roofline informs the plan's cost
// but is NEVER read by -bcir-verify as a legality verdict -- the conv op is NOT erased and never
// gates legality (the same two-truth quarantine the sibling cost passes keep).
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIR/BCIRPasses.h"
#include "BCIRPassSupport.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>

using namespace mlir;

namespace bcir {
namespace {

// ceil(a / b) for positive b (mirrors conv.py / matmul.py's (x + d - 1)//d / math.ceil and the
// sibling cost-pass ceilDiv helpers).
static int64_t ceilDiv(int64_t a, int64_t b) {
  return a / b + (a % b != 0);
}

struct GemConvCostPass : public PassWrapper<GemConvCostPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemConvCostPass)

  StringRef getArgument() const final { return "bcir-gem-conv-cost"; }
  StringRef getDescription() const final {
    return "Recompute the G7 conv roofline (port of conv.py::cost_of -- a conv IS a structured "
           "matmul priced through matmul.cost_of on its im2col gemm dims) for each gem.conv op and "
           "parity-check the declared compute/mem/bottleneck; annotate kbcir.* INFORMS-ONLY (never "
           "a legality verdict).";
  }

  void runOnOperation() override {
    SmallVector<GEMConvOp> ops;
    getOperation()->walk([&](GEMConvOp cv) { ops.push_back(cv); });
    bool ok = true;
    for (GEMConvOp cv : ops)
      ok &= priceOne(cv);
    if (!ok)
      signalPassFailure();
  }

  // Recompute the roofline cost from the op's equivalent im2col gemm dims + tiling through the
  // pinned x86-avx512 reference constants (the conv-is-a-structured-matmul pricing -- identical to
  // conv.cost_of's matmul.cost_of call), cross-check the declared compute/mem/bottleneck, and
  // annotate the recomputed roofline. Returns false on a parity violation. (The op is NOT erased --
  // this is a cost producer, not a lowering.)
  bool priceOne(GEMConvOp op) {
    MLIRContext *ctx = op.getContext();
    Builder ab(ctx);

    // The pinned TargetProfile.x86_avx512() reference constants (conv.py::cost_of -> matmul.cost_of
    // reads these off the target; pinned here for CI-host-independence, EXACTLY as the sibling cost
    // passes + their *_law_parity.py tests pin x86-avx512): vector_width = max(1,8,16) = 16,
    // fma = true, mem_unit = 1, mem_channels = 4. (conv/matmul cost_of use none of cacheline/
    // elem_bytes -- those only gate fits_cache, which is not a parity-checked roofline term.)
    const int64_t kVectorWidth = 16;
    const int64_t kFma = 2; // (2 if fma else 1); avx512 has fma -> 2
    const int64_t kMemUnit = 1;
    const int64_t kMemChannels = 4;

    // The equivalent im2col gemm dims the conv is priced through (M = out_h*out_w = output pixels,
    // N = out_c, K = in_c*kh*kw = the taps) + the resolved inner tile. The op carries both (the
    // verifier pins them internally consistent: gemm dims == the derived structured-matmul dims;
    // direct -> tile == the whole gemm). cost_of prices matmul.cost_of at exactly (gemm dims, tile).
    int64_t M = static_cast<int64_t>(op.getGemmM());
    int64_t N = static_cast<int64_t>(op.getGemmN());
    int64_t K = static_cast<int64_t>(op.getGemmK());
    int64_t tm = std::max<int64_t>(1, static_cast<int64_t>(op.getTileM()));
    int64_t tn = std::max<int64_t>(1, static_cast<int64_t>(op.getTileN()));
    int64_t tk = std::max<int64_t>(1, static_cast<int64_t>(op.getTileK()));

    // --- recompute the analytic roofline (identical to conv.py::cost_of == matmul.cost_of on the
    // im2col gemm dims) ----------------------------------------------------------------------------
    // Computed in int64 -- the cost domain the GEMConvOp attrs themselves are declared in (I64Attr) and
    // that the verifier + every sibling cost pass share. A conv whose im2col M*N*K (or ~3*M*N*K traffic)
    // would exceed int64 -- a gemm dim past ~1.45M, i.e. multi-terabyte operands no real conv approaches
    // -- is outside this representable domain anyway (its own declared cost could not fit i64); such a
    // shape would trip the parity gate (a loud spurious error), never annotate a silently-wrapped cost.
    int64_t macs, traffic, workingSet;
    if (!checkedGemmDerived(M, N, K, tm, tn, tk, macs, traffic, workingSet)) {
      op.emitError("bcir-gem-conv-cost: derived roofline cost exceeds signed 64-bit range");
      return false;
    }
    int64_t thr = std::max<int64_t>(1, kVectorWidth * kFma);    // the FLOP-throughput divisor
    int64_t compute = ceilDiv(macs, thr);                       // M*N*K MACs / (vector_width * fma)
    int64_t bw = std::max<int64_t>(1, kMemUnit * kMemChannels); // the bandwidth divisor
    int64_t mem = ceilDiv(traffic, bw);                         // bytes streamed / bandwidth
    int64_t bottleneck = std::max(compute, mem); // max,+ : the binding roofline resource

    // --- cross-check the declared roofline (the dual-rail parity gate) ------------------
    if (compute != static_cast<int64_t>(op.getComputeCost())) {
      op.emitError("bcir-gem-conv-cost: compute_cost ")
          << static_cast<int64_t>(op.getComputeCost()) << " != the recomputed roofline compute "
          << compute << " (ceilDiv(gemm_m*gemm_n*gemm_k, vector_width*fma))";
      return false;
    }
    if (mem != static_cast<int64_t>(op.getMemCost())) {
      op.emitError("bcir-gem-conv-cost: mem_cost ")
          << static_cast<int64_t>(op.getMemCost()) << " != the recomputed roofline mem " << mem
          << " (ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels))";
      return false;
    }
    if (bottleneck != static_cast<int64_t>(op.getBottleneck())) {
      op.emitError("bcir-gem-conv-cost: bottleneck ")
          << static_cast<int64_t>(op.getBottleneck()) << " != max(compute, mem) " << bottleneck;
      return false;
    }

    // --- annotate the recomputed roofline (the cross-pass dual-rail record) -------------
    // INFORMS-ONLY: the recomputed roofline informs the plan's cost but never gates legality
    // (-bcir-verify never reads these).
    op->setAttr("kbcir.compute_cost", ab.getI64IntegerAttr(compute));
    op->setAttr("kbcir.mem_cost", ab.getI64IntegerAttr(mem));
    op->setAttr("kbcir.bottleneck", ab.getI64IntegerAttr(bottleneck));
    op->setAttr("kbcir.informs_only", ab.getBoolAttr(true)); // never a legality verdict
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> createGemConvCostPass() {
  return std::make_unique<GemConvCostPass>();
}

} // namespace bcir
