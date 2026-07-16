//===- BCIRLowerGemMatmulPass.cpp - the -bcir-lower-gem-matmul lowering -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
// Lowers each `bcir.gem.matmul` plan record (the K_BCIR-chosen tiling/loop-order
// realization, B1) into its CONCRETE TILED REALIZATION: one `bcir.gem.block`
// descriptor per tile of the tiled iteration space, emitted in the plan's
// loop_order. This makes the tiling EXPLICIT -- the matmul op carries the chosen
// tile extents and loop order as attributes; this pass enumerates the tiles that
// choice induces. It is a genuine lowering: the `gem.matmul` is ERASED after its
// tiles are emitted, leaving the tile-block sequence in its place.
//
// LOWERING SEMANTICS (documented + verified by mlir/test/passes/lower_gem_matmul.mlir):
//   * Iteration space: tiles over (M, N, K) with steps (tile_m, tile_n, tile_k);
//     ceil(M/tm) * ceil(N/tn) * ceil(K/tk) tiles total.
//   * Loop order: the three tile loops nest in the plan's loop_order --
//       ijk = i outer, j mid, k inner;  ikj = i, k, j;  jik = j, i, k.
//     Blocks are emitted in that nesting order (the outermost index varies slowest).
//   * sym_name: <matmul-name>_t<idx>, idx = 0-based emission order (e.g. @mm0_t0, @mm0_t1).
//   * base: the linearized origin of the tile in the row-major M x N output C =
//       i0 * N + j0  (i0/j0 are the tile's row/col origins; k does NOT move base --
//       a tile accumulates over k into the same C block).
//   * count: the tile's C element count = min(tm, M - i0) * min(tn, N - j0) (clamps
//     the ragged boundary tiles; independent of k).
//   * strideA = K (A row stride, A is M x K row-major), strideB = N (B row stride,
//     B is K x N row-major), strideD = N (C row stride, C is M x N row-major).
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRPassSupport.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <string>

using namespace mlir;

namespace bcir {
namespace {

// ceil(a / b) for positive b (the tile-count along one dimension).
static int64_t ceilDiv(int64_t a, int64_t b) {
  return a / b + (a % b != 0);
}

struct LowerGemMatmulPass
    : public PassWrapper<LowerGemMatmulPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGemMatmulPass)

  StringRef getArgument() const final { return "bcir-lower-gem-matmul"; }
  StringRef getDescription() const final {
    return "Lower each gem.matmul plan into its concrete tiled realization "
           "(one gem.block per tile, in loop_order; erases the matmul).";
  }

  void runOnOperation() override {
    // Collect first, then rewrite: lowering mutates the IR (creates blocks, erases
    // the matmul), so we must not be walking the op list as we change it.
    SmallVector<GEMMatmulOp> matmuls;
    getOperation()->walk([&](GEMMatmulOp mm) { matmuls.push_back(mm); });

    bool ok = true;
    for (GEMMatmulOp mm : matmuls)
      ok &= lowerOne(mm);
    if (!ok)
      signalPassFailure();
  }

  // Emit the tile-block sequence for one matmul, then erase the matmul.
  bool lowerOne(GEMMatmulOp mm) {
    const int64_t M = static_cast<int64_t>(mm.getM());
    const int64_t N = static_cast<int64_t>(mm.getN());
    const int64_t K = static_cast<int64_t>(mm.getK());
    const int64_t tm = static_cast<int64_t>(mm.getTileM());
    const int64_t tn = static_cast<int64_t>(mm.getTileN());
    const int64_t tk = static_cast<int64_t>(mm.getTileK());
    const StringRef loopOrder = mm.getLoopOrder();
    const StringRef name = mm.getSymName();

    int64_t tileCount;
    if (!checkedTileCount(M, N, K, tm, tn, tk, tileCount) ||
        tileCount > kMaxMaterializedBlocksPerOp) {
      mm.emitError("bcir-lower-gem-matmul: tile expansion is invalid or exceeds the ")
          << kMaxMaterializedBlocksPerOp << "-block safety limit";
      return false;
    }

    // Insert the emitted blocks right where the matmul lives (same parent region),
    // immediately AFTER it, so the IR order reads matmul-replaced-by-its-tiles.
    OpBuilder builder(mm);
    builder.setInsertionPointAfter(mm);
    const Location loc = mm.getLoc();
    MLIRContext *ctx = mm.getContext();

    const int64_t nI = ceilDiv(M, tm);  // tile loop trip counts
    const int64_t nJ = ceilDiv(N, tn);
    const int64_t nK = ceilDiv(K, tk);

    // The three nested tile loops, ordered by loop_order. Each entry is a tile
    // origin (i0, j0); we walk the loop nest in order and emit one block per
    // innermost iteration. The loop_order only changes the ENUMERATION order of
    // the tiles -- the per-tile descriptor (base/count/strides) is loop-order
    // invariant (every realization reproduces the same result; LangRef Sec. 1).
    int64_t idx = 0;
    auto emitTile = [&](int64_t i, int64_t j) {
      const int64_t i0 = i * tm;
      const int64_t j0 = j * tn;
      const int64_t rows = std::min(tm, M - i0);
      const int64_t cols = std::min(tn, N - j0);
      const int64_t base = i0 * N + j0;   // row-major C origin
      const int64_t count = rows * cols;  // C-tile element count

      std::string sym = (name + "_t" + Twine(idx)).str();
      builder.create<GEMBlockOp>(
          loc,
          /*sym_name=*/StringAttr::get(ctx, sym),
          /*base=*/builder.getI64IntegerAttr(base),
          /*count=*/builder.getI64IntegerAttr(count),
          /*strideA=*/builder.getI64IntegerAttr(K),
          /*strideB=*/builder.getI64IntegerAttr(N),
          /*strideD=*/builder.getI64IntegerAttr(N));
      ++idx;
    };

    // Walk the tile nest in the plan's loop_order. ijk = i outer, j mid, k inner;
    // ikj = i, k, j; jik = j, i, k. (The verifier guarantees loop_order is one of
    // these three; default to ijk defensively.)
    if (loopOrder == "ijk") {
      for (int64_t i = 0; i < nI; ++i)
        for (int64_t j = 0; j < nJ; ++j)
          for (int64_t k = 0; k < nK; ++k)
            emitTile(i, j);
    } else if (loopOrder == "ikj") {
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
      // Defensive fallback (verifier should have rejected anything else): ijk.
      for (int64_t i = 0; i < nI; ++i)
        for (int64_t j = 0; j < nJ; ++j)
          for (int64_t k = 0; k < nK; ++k)
            emitTile(i, j);
    }

    // Genuine lowering: the plan record is consumed -- erase it, leaving the
    // concrete tile descriptors in its place.
    mm.erase();
    return true;
  }
};

}  // namespace

std::unique_ptr<Pass> createLowerGemMatmulPass() {
  return std::make_unique<LowerGemMatmulPass>();
}

}  // namespace bcir
