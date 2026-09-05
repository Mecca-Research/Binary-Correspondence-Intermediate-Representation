//===- BCIRLowerGemConvPass.cpp - the -bcir-lower-gem-conv lowering -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
// Lowers each `bcir.gem.conv` plan record (the K_BCIR-chosen direct|im2col
// realization of a 2-D single-group convolution, G7) into its CONCRETE
// REALIZATION: one `bcir.gem.block` descriptor per tile of the EQUIVALENT im2col
// gemm's tiled iteration space, emitted in the plan's loop_order. A conv IS a
// STRUCTURED MATMUL -- the im2col reshaping turns it into the gemm
// C[out_pix x C_out] = patches[out_pix x (C_in*Kh*Kw)] @ W[(C_in*Kh*Kw) x C_out] --
// so this pass lowers the conv through the SAME tiled-gem.block emission the
// -bcir-lower-gem-matmul pass uses on that gemm (NOT a bespoke conv loop nest). It
// is a genuine lowering: the `gem.conv` is ERASED after its tiles are emitted.
//
// THE DUAL-RAIL PARITY THIS PASS ENFORCES (mirrors bcir/kbcir/conv.py exactly,
// verified by mlir/test/passes/lower_gem_conv.mlir):
//   * It prices the conv THROUGH THE EXISTING matmul roofline (conv.cost_of): the
//     im2col gemm's (compute, mem) is exactly matmul.cost_of at the chosen tile;
//     the direct strategy is the UNTILED gemm (whole gemm_m/n/k). The host-pinned
//     compute_cost/mem_cost ride the op (the oracle plans them with the pinned
//     target, which differs per arch/CI); this pass CROSS-CHECKS the host-
//     independent invariant the cost model rests on -- bottleneck == max(compute,
//     mem) (the max,+ step) -- exactly as -bcir-lower-gem-activation does. The
//     chosen strategy + cost on the op MUST equal the oracle's plan_conv/cost_of
//     output (that is the gate that matters; re-asserted host-agnostically in
//     bcir/tests/test_conv_law_parity.py, which pins the oracle target x86-avx512).
//   * It reproduces plan_conv's STRATEGY CHOICE as the realization: an `im2col`
//     conv lowers the gemm in the chosen tile (the blocked-gemm reuse); a `direct`
//     conv lowers the UNTILED gemm (one whole-gemm block). The op carries the
//     K_BCIR-chosen strategy + tile; this pass enumerates exactly the tiles that
//     choice induces (the strategy is decided in the oracle, pinned on the op).
//   * It reproduces the R17 accuracy axis: a conv's MACs are exact, so the only
//     certified error of a QUANTIZED realization is the Q8 bridge step (R17-bounded
//     at 1 ULP); a dense conv carries acc_bound 0.
//
// EMITTED REALIZATION (one gem.block per im2col-gemm tile; documented + FileCheck'd):
//   * Iteration space: tiles over (gemm_m, gemm_n, gemm_k) with steps (tile_m,
//     tile_n, tile_k); ceil(M/tm) * ceil(N/tn) * ceil(K/tk) tiles total.
//   * Loop order: the three tile loops nest in the plan's loop_order (ijk/ikj/jik),
//     exactly as -bcir-lower-gem-matmul.
//   * sym_name: <conv-name>_t<idx>, idx = 0-based emission order (e.g. @cv0_t0).
//   * base: the linearized origin of the tile in the row-major M x N gemm output =
//     i0 * gemm_n + j0 (k does NOT move base -- a tile accumulates over k).
//   * count: the tile's element count = min(tm, M-i0) * min(tn, N-j0).
//   * strideA = gemm_k (patch-row stride, P is M x K row-major), strideB = gemm_n
//     (W-row stride, Wm is K x N row-major), strideD = gemm_n (output row stride).
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
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <string>

using namespace mlir;

namespace bcir {
namespace {

// ceil(a / b) for positive b (the tile-count along one im2col-gemm dimension).
static int64_t ceilDiv(int64_t a, int64_t b) {
  return a / b + (a % b != 0);
}

struct LowerGemConvPass : public PassWrapper<LowerGemConvPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGemConvPass)

  StringRef getArgument() const final { return "bcir-lower-gem-conv"; }
  StringRef getDescription() const final {
    return "Lower each gem.conv plan into its concrete realization (the im2col "
           "gemm tiled into one gem.block per tile, in loop_order; recomputes the "
           "K_BCIR roofline parity + the R17 verdict; erases the conv).";
  }

  void runOnOperation() override {
    // Collect first, then rewrite: lowering mutates the IR (creates blocks, erases the
    // conv), so we must not walk the op list while we change it.
    SmallVector<GEMConvOp> convs;
    getOperation()->walk([&](GEMConvOp c) { convs.push_back(c); });
    bool ok = true;
    for (GEMConvOp c : convs)
      ok &= lowerOne(c);
    if (!ok)
      signalPassFailure();
  }

  // Recompute the host-independent roofline + R17 invariants, emit the im2col-gemm
  // tile-block sequence for one conv, then erase the conv. Returns false on a parity
  // violation.
  bool lowerOne(GEMConvOp conv) {
    MLIRContext *ctx = conv.getContext();
    Builder ab(ctx);
    const StringRef name = conv.getSymName();
    const StringRef strategy = conv.getStrategy();
    const StringRef loopOrder = conv.getLoopOrder();

    // The equivalent im2col gemm dims (a conv IS a structured matmul) + the chosen tile.
    const int64_t M = static_cast<int64_t>(conv.getGemmM());
    const int64_t N = static_cast<int64_t>(conv.getGemmN());
    const int64_t K = static_cast<int64_t>(conv.getGemmK());
    const int64_t tm = static_cast<int64_t>(conv.getTileM());
    const int64_t tn = static_cast<int64_t>(conv.getTileN());
    const int64_t tk = static_cast<int64_t>(conv.getTileK());
    int64_t tileCount;
    if (!checkedTileCount(M, N, K, tm, tn, tk, tileCount) ||
        tileCount > kMaxMaterializedBlocksPerOp) {
      conv.emitError("bcir-lower-gem-conv: tile expansion is invalid or exceeds the ")
          << kMaxMaterializedBlocksPerOp << "-block safety limit";
      return false;
    }
    // S0-8 / row 13 (checked arithmetic): every tile's origin `i0*N + j0` and element count
    // `rows*cols` lie within the M x N output, so the one bound that matters is the output
    // element count itself. The op verifier refuses a conv whose M*N does not fit the signed
    // 64-bit wire domain; a programmatically built op is held to the same bound here, and the
    // per-tile arithmetic below stays checked -- a verifier-legal one-tile conv (M = 2^62,
    // N = 4) used to wrap its count to 0 (lower_gem_conv_overflow.mlir).
    int64_t outElems;
    if (!checkedMulNonnegative(M, N, outElems)) {
      conv.emitError("bcir-lower-gem-conv: output element count M*N exceeds signed 64-bit range");
      return false;
    }

    // --- cross-check the host-INDEPENDENT roofline invariant the cost model rests on --
    // The host-pinned compute_cost/mem_cost ride the op (the oracle prices them through
    // matmul.cost_of with the pinned target, so we do not re-derive the arch-specific
    // thr/bw here). We cross-check bottleneck == max(compute, mem) (the max,+ step), the
    // same recompute-and-cross-check -bcir-lower-gem-activation applies.
    int64_t compute = static_cast<int64_t>(conv.getComputeCost());
    int64_t mem = static_cast<int64_t>(conv.getMemCost());
    int64_t bottleneck = static_cast<int64_t>(conv.getBottleneck());
    int64_t mx = std::max(compute, mem);
    if (bottleneck != mx) {
      conv.emitError("bcir-lower-gem-conv: bottleneck ")
          << bottleneck << " != max(compute_cost, mem_cost) " << mx
          << " (the max,+ roofline parity)";
      return false;
    }
    // The direct stream is the UNTILED gemm (it can not block the reduction); reproduce
    // plan_conv's strategy invariant (conv.cost_of's direct branch == matmul.cost_of at
    // tile = whole dims).
    if (strategy == "direct" && (tm != M || tn != N || tk != K)) {
      conv.emitError("bcir-lower-gem-conv: the direct strategy is the UNTILED gemm -- "
                     "tile must be the whole ")
          << M << "x" << N << "x" << K << " (got " << tm << "x" << tn << "x" << tk << ")";
      return false;
    }

    // --- the R17 accuracy axis (mirrors conv.cost_vector) -----------------------------
    int64_t quantBits = static_cast<int64_t>(conv.getQuantBits());
    int64_t accBound = quantBits > 0 ? 1 : 0; // the Q8-bridge bound iff quantized (1 ULP), else 0
    if (accBound != static_cast<int64_t>(conv.getAccBound())) {
      conv.emitError("bcir-lower-gem-conv: acc_bound ")
          << static_cast<int64_t>(conv.getAccBound()) << " != the R17 Q8-bridge bound " << accBound
          << " (0 dense / 1 ULP quantized)";
      return false;
    }

    // Insert the emitted blocks right where the conv lives, immediately AFTER it.
    OpBuilder builder(conv);
    builder.setInsertionPointAfter(conv);
    const Location loc = conv.getLoc();

    const int64_t nI = ceilDiv(M, tm); // tile loop trip counts over the im2col gemm
    const int64_t nJ = ceilDiv(N, tn);
    const int64_t nK = ceilDiv(K, tk);

    // The per-tile descriptor (base/count/strides) is loop-order invariant (every
    // realization reproduces the same conv result); the loop_order only changes the
    // ENUMERATION order. strideA = K (P is M x K row-major), strideB = N (Wm is K x N),
    // strideD = N (the output is M x N) -- the im2col gemm strides, exactly as
    // -bcir-lower-gem-matmul.
    int64_t idx = 0;
    bool overflow = false;
    auto emitTile = [&](int64_t i, int64_t j) {
      int64_t i0, j0, rowBase, base, count;
      if (!checkedMulNonnegative(i, tm, i0) || !checkedMulNonnegative(j, tn, j0) ||
          !checkedMulNonnegative(i0, N, rowBase) || !checkedAddNonnegative(rowBase, j0, base) ||
          !checkedMulNonnegative(std::min(tm, M - i0), std::min(tn, N - j0), count)) {
        overflow = true; // unreachable under the M*N bound above; never silent if it is not
        return;
      }

      std::string sym = (name + "_t" + Twine(idx)).str();
      GEMBlockOp::create(builder, loc,
                         /*sym_name=*/StringAttr::get(ctx, sym),
                         /*base=*/builder.getI64IntegerAttr(base),
                         /*count=*/builder.getI64IntegerAttr(count),
                         /*strideA=*/builder.getI64IntegerAttr(K),
                         /*strideB=*/builder.getI64IntegerAttr(N),
                         /*strideD=*/builder.getI64IntegerAttr(N));
      ++idx;
    };

    // Walk the tile nest in the plan's loop_order (ijk = i outer, j mid, k inner; ikj =
    // i, k, j; jik = j, i, k), identical to -bcir-lower-gem-matmul. (The verifier
    // guarantees loop_order is one of these three; default to ijk defensively.)
    if (loopOrder == "ikj") {
      for (int64_t i = 0; i < nI; ++i)
        for (int64_t k = 0; k < nK; ++k)
          for (int64_t j = 0; j < nJ; ++j)
            emitTile(i, j);
    } else if (loopOrder == "jik") {
      for (int64_t j = 0; j < nJ; ++j)
        for (int64_t i = 0; i < nI; ++i)
          for (int64_t k = 0; k < nK; ++k)
            emitTile(i, j);
    } else {
      for (int64_t i = 0; i < nI; ++i)
        for (int64_t j = 0; j < nJ; ++j)
          for (int64_t k = 0; k < nK; ++k)
            emitTile(i, j);
    }

    if (overflow) {
      conv.emitError("bcir-lower-gem-conv: tile origin/element count exceeds signed 64-bit range");
      return false;
    }

    // Annotate the recomputed K_BCIR plan (strategy + the priced roofline + the R17
    // verdict) on the FIRST emitted tile block, as the cross-pass dual-rail record.
    if (idx > 0) {
      getOperation()->walk([&](GEMBlockOp b) {
        if (b.getSymName() == (name + "_t0").str()) {
          b->setAttr("kbcir.strategy", ab.getStringAttr(strategy));
          b->setAttr("kbcir.gemm_m", ab.getI64IntegerAttr(M));
          b->setAttr("kbcir.gemm_n", ab.getI64IntegerAttr(N));
          b->setAttr("kbcir.gemm_k", ab.getI64IntegerAttr(K));
          b->setAttr("kbcir.loop_order", ab.getStringAttr(loopOrder));
          b->setAttr("kbcir.compute_cost", ab.getI64IntegerAttr(compute));
          b->setAttr("kbcir.mem_cost", ab.getI64IntegerAttr(mem));
          b->setAttr("kbcir.bottleneck", ab.getI64IntegerAttr(bottleneck));
          b->setAttr("kbcir.acc_bound", ab.getI64IntegerAttr(accBound));
        }
      });
    }

    // Genuine lowering: the plan record is consumed -- erase it.
    conv.erase();
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> createLowerGemConvPass() {
  return std::make_unique<LowerGemConvPass>();
}

} // namespace bcir
