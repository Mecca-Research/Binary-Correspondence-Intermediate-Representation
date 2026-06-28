//===- BCIRLowerGemMatmulBufferPass.cpp - -bcir-lower-gem-matmul-buffer -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library. Shared helpers live in
// BCIRPassSupport.h; registration in BCIRPasses.cpp. C++23.
//
// Lowers each `bcir.gem.matmul_buffer` (real SSA memref operands A/B/C plus a tile
// plan) into a CONCRETE TILED scf.for LOOP NEST computing `C += A*B` with
// memref.load/store + arith.mulf/addf. Unlike -bcir-lower-gem-matmul (which emits
// attribute-only gem.block descriptors), this emits executable-style structured IR.
// It is a genuine lowering: the matmul_buffer op is ERASED after the nest is emitted.
//
// LOWERING SEMANTICS (documented + verified by
// mlir/test/passes/lower_gem_matmul_buffer.mlir):
//   The canonical tiled matmul, three OUTER tiled loops permuted by loop_order:
//     scf.for i = 0 to M step tile_m {            (ijk: i outer, j mid, k inner)
//       scf.for j = 0 to N step tile_n {          (ikj: i, k, j)
//         scf.for k = 0 to K step tile_k {        (jik: j, i, k)
//           iU = min(i+tile_m, M); jU = min(j+tile_n, N); kU = min(k+tile_k, K)
//           scf.for ii = i to iU step 1 {         (point loops always ii -> jj -> kk)
//             scf.for jj = j to jU step 1 {
//               scf.for kk = k to kU step 1 {
//                 a = A[ii, kk]; b = B[kk, jj]; c = C[ii, jj]
//                 C[ii, jj] = c + a*b
//   }}}}}}
//   M/N/K come from the operands' static memref shapes (verifier guarantees them).
//   The clamps make ragged-boundary tiles correct; the in-place accumulation reads
//   and writes C[ii,jj] each kk step (no tile-private accumulator -- simplest correct
//   form; matches the structural test). The outer-loop PERMUTATION is observable in
//   the emitted nest; the point loops + fma body are loop-order invariant.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRPassSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

#include <array>

using namespace mlir;

namespace bcir {
namespace {

struct LowerGemMatmulBufferPass
    : public PassWrapper<LowerGemMatmulBufferPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGemMatmulBufferPass)

  StringRef getArgument() const final {
    return "bcir-lower-gem-matmul-buffer";
  }
  StringRef getDescription() const final {
    return "Lower each gem.matmul_buffer (SSA memref A/B/C + tile plan) into a "
           "concrete tiled scf.for nest computing C += A*B (load/store + "
           "mulf/addf); erases the op.";
  }

  // The pass BUILDS ops in these dialects, so they must be loaded into the context
  // before it runs (the registry only makes them available; this loads them).
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    // Collect first, then rewrite: lowering creates loops and erases the op, so
    // we must not be walking the op list while mutating it.
    SmallVector<GEMMatmulBufferOp> ops;
    getOperation()->walk([&](GEMMatmulBufferOp op) { ops.push_back(op); });
    for (GEMMatmulBufferOp op : ops)
      lowerOne(op);
  }

  void lowerOne(GEMMatmulBufferOp op) {
    OpBuilder builder(op);
    const Location loc = op.getLoc();

    Value A = op.getA(), B = op.getB(), C = op.getC();
    auto cTy = cast<MemRefType>(C.getType());
    auto aTy = cast<MemRefType>(A.getType());
    auto bTy = cast<MemRefType>(B.getType());
    // M/N/K from static shapes (verifier guarantees rank-2 f32 static memrefs and
    // that the dims agree across A/B/C).
    const int64_t M = aTy.getDimSize(0);
    const int64_t K = aTy.getDimSize(1);
    const int64_t N = bTy.getDimSize(1);
    (void)cTy;
    const int64_t tm = static_cast<int64_t>(op.getTileM());
    const int64_t tn = static_cast<int64_t>(op.getTileN());
    const int64_t tk = static_cast<int64_t>(op.getTileK());
    const StringRef loopOrder = op.getLoopOrder();

    auto cst = [&](int64_t v) -> Value {
      return builder.create<arith::ConstantIndexOp>(loc, v);
    };
    Value c0 = cst(0), c1 = cst(1);
    Value cM = cst(M), cN = cst(N), cK = cst(K);
    Value cTm = cst(tm), cTn = cst(tn), cTk = cst(tk);

    // Describe each tiled axis: its lower bound origin, full extent (for the clamp),
    // and tile step. We build the three outer loops in the permutation given by
    // loop_order; the body (clamps + point loops + fma) is emitted at the innermost
    // outer loop regardless of the permutation.
    struct Axis {
      char id;        // 'i' | 'j' | 'k'
      Value extent;   // M / N / K
      Value step;     // tile_m / tile_n / tile_k
    };
    Axis axI{'i', cM, cTm};
    Axis axJ{'j', cN, cTn};
    Axis axK{'k', cK, cTk};

    // loop_order permutation of the three OUTER tiled loops. The verifier restricts
    // loop_order to {ijk, ikj, jik}; default to ijk defensively.
    std::array<Axis, 3> order{axI, axJ, axK};
    if (loopOrder == "ijk")
      order = {axI, axJ, axK};
    else if (loopOrder == "ikj")
      order = {axI, axK, axJ};
    else if (loopOrder == "jik")
      order = {axJ, axI, axK};

    // Build the three nested outer tiled loops. Track the per-axis tile origin
    // (induction var) so the body can index by 'i'/'j'/'k' regardless of nest order.
    Value tileLB[3] = {nullptr, nullptr, nullptr};  // indexed by 0=i,1=j,2=k
    auto axisIndex = [](char id) { return id == 'i' ? 0 : (id == 'j' ? 1 : 2); };

    for (const Axis &ax : order) {
      auto forOp = builder.create<scf::ForOp>(loc, c0, ax.extent, ax.step);
      tileLB[axisIndex(ax.id)] = forOp.getInductionVar();
      // Insert subsequent ops (and the next loop) into this loop's body, before
      // the implicit scf.yield terminator.
      builder.setInsertionPointToStart(forOp.getBody());
    }

    Value iLB = tileLB[0], jLB = tileLB[1], kLB = tileLB[2];

    // Ragged-boundary clamps: upper bound of each point loop is min(origin + tile, dim).
    auto clamp = [&](Value lb, Value step, Value dim) -> Value {
      Value sum = builder.create<arith::AddIOp>(loc, lb, step);
      return builder.create<arith::MinSIOp>(loc, sum, dim);
    };
    Value iU = clamp(iLB, cTm, cM);
    Value jU = clamp(jLB, cTn, cN);
    Value kU = clamp(kLB, cTk, cK);

    // Point loops: ii -> jj -> kk, step 1. The fma body lives at the innermost.
    auto iiFor = builder.create<scf::ForOp>(loc, iLB, iU, c1);
    builder.setInsertionPointToStart(iiFor.getBody());
    Value ii = iiFor.getInductionVar();

    auto jjFor = builder.create<scf::ForOp>(loc, jLB, jU, c1);
    builder.setInsertionPointToStart(jjFor.getBody());
    Value jj = jjFor.getInductionVar();

    auto kkFor = builder.create<scf::ForOp>(loc, kLB, kU, c1);
    builder.setInsertionPointToStart(kkFor.getBody());
    Value kk = kkFor.getInductionVar();

    // C[ii,jj] += A[ii,kk] * B[kk,jj]
    Value a = builder.create<memref::LoadOp>(loc, A, ValueRange{ii, kk});
    Value b = builder.create<memref::LoadOp>(loc, B, ValueRange{kk, jj});
    Value c = builder.create<memref::LoadOp>(loc, C, ValueRange{ii, jj});
    Value prod = builder.create<arith::MulFOp>(loc, a, b);
    Value acc = builder.create<arith::AddFOp>(loc, c, prod);
    builder.create<memref::StoreOp>(loc, acc, C, ValueRange{ii, jj});

    // Genuine lowering: the plan op is consumed.
    op.erase();
  }
};

}  // namespace

std::unique_ptr<Pass> createLowerGemMatmulBufferPass() {
  return std::make_unique<LowerGemMatmulBufferPass>();
}

}  // namespace bcir
