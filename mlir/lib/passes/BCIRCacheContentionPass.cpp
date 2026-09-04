//===- BCIRCacheContentionPass.cpp - the -bcir-cache-contention cost signal -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
// Recomputes the G4 cache-line/memory-bank-conflict CONTENTION cost SIGNAL for each
// `bcir.gem.contention` carrier op (the law-rail port of bcir/kbcir/cache_predict.py::
// CachePredictor.contention_q8) and annotates the recomputed Q8 terms + the integer
// CONTENTION cost on the op. Unlike the lower-gem-* passes this is a COST PRODUCER, not
// a lowering: the op is NOT erased -- the recomputed signal is annotated as kbcir.*
// attrs (the cross-pass dual-rail record), exactly as the cost/plan/overlap analysis
// passes annotate rather than rewrite.
//
// THE FROZEN ANALYTIC MODEL THIS PASS REPRODUCES (mirrors cache_predict.py exactly,
// verified by mlir/test/passes/cache_contention.mlir + bcir/tests/test_cache_contention_
// law_parity.py):
//   * cache-line-utilization WASTE: a line holds epl = cacheline/elem_bytes useful
//     elements. A unit-stride sweep packs all epl into every line (waste 0); a strided
//     access of step s touches reach = min(s, epl) distinct lines per useful element, so
//     line_waste_q8 = (reach - 1) * 256 (saturates at epl-1, a full gather).
//   * bank conflicts: interleaved memory has `banks` (== TargetProfile.mem_channels)
//     banks; a stride s reaches banks/gcd(s, banks) distinct banks and SERIALIZES by the
//     shared factor g = gcd(s, banks), so bank_conflict_q8 = (g - 1) * 256 (0 == reaches
//     all banks; maximal at a stride == #banks -> one bank).
//   * contention_q8 = waste + conflict (unit Q8 weights, both x1.0); contention_cost =
//     (contention_q8 * count) >> 8 (the per-element Q8 scaled by the access count).
//
// THE TWO-TRUTH / QUARANTINE THIS PASS PRESERVES (the non-negotiable G4 discipline): the
// recomputed signal is INFORMS-ONLY. It rides the CONTENTION cost axis and is annotated
// as kbcir.* attrs that only INFORM plan ranking; it NEVER produces a legality verdict
// and is NEVER read by -bcir-verify (the R1-R23 law set reads only the relevant structural data;
// completeness/non-negativity -- never CONTENTION). A conflict-prone access and a
// conflict-free one are EXACTLY as legal; this pass only moves the CONTENTION dimension.
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
#include <numeric>

using namespace mlir;

namespace bcir {
namespace {

struct CacheContentionPass : public PassWrapper<CacheContentionPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CacheContentionPass)

  StringRef getArgument() const final { return "bcir-cache-contention"; }
  StringRef getDescription() const final {
    return "Recompute the G4 cache-line/bank-conflict CONTENTION cost signal for each "
           "gem.contention op (port of cache_predict.py) and annotate the Q8 terms + the "
           "integer cost; INFORMS-ONLY (the CONTENTION axis never gates legality).";
  }

  void runOnOperation() override {
    SmallVector<GEMContentionOp> ops;
    getOperation()->walk([&](GEMContentionOp c) { ops.push_back(c); });
    bool ok = true;
    for (GEMContentionOp c : ops)
      ok &= priceOne(c);
    if (!ok)
      signalPassFailure();
  }

  // Recompute the frozen analytic signal from the op's access shape + frozen params, cross-
  // check the declared fields, and annotate the recomputed signal. Returns false on a parity
  // violation. (The op is NOT erased -- this is a cost producer, not a lowering.)
  bool priceOne(GEMContentionOp op) {
    MLIRContext *ctx = op.getContext();
    Builder ab(ctx);

    const int64_t Q8 = 256;
    int64_t stride = std::max<int64_t>(1, static_cast<int64_t>(op.getStride()));
    int64_t eb = std::max<int64_t>(1, static_cast<int64_t>(op.getElemBytes()));
    int64_t count = std::max<int64_t>(1, static_cast<int64_t>(op.getCount()));
    int64_t cl = std::max<int64_t>(1, static_cast<int64_t>(op.getCacheline()));
    int64_t banks = std::max<int64_t>(1, static_cast<int64_t>(op.getBanks()));

    // --- the frozen analytic model (identical to CachePredictor.contention_q8) ---------
    int64_t epl = std::max<int64_t>(1, cl / eb);    // useful elements per cache line
    int64_t reach = std::min<int64_t>(stride, epl); // distinct lines per useful element
    int64_t waste, conflict, cont, scaled;
    if (!checkedMulNonnegative(reach - 1, Q8, waste)) {
      op.emitError("bcir-cache-contention: derived signal exceeds signed 64-bit range");
      return false;
    }
    int64_t g = std::gcd(stride, banks); // the shared factor: serialization degree
    if (!checkedMulNonnegative(g - 1, Q8, conflict) ||
        !checkedAddNonnegative(waste, conflict, cont) ||
        !checkedMulNonnegative(cont, count, scaled)) {
      op.emitError("bcir-cache-contention: derived signal exceeds signed 64-bit range");
      return false;
    }
    int64_t cost = scaled >> 8; // the per-element Q8 scaled by count

    // --- cross-check the declared signal (the dual-rail parity gate) -------------------
    if (waste != static_cast<int64_t>(op.getLineWasteQ8())) {
      op.emitError("bcir-cache-contention: line_waste_q8 ")
          << static_cast<int64_t>(op.getLineWasteQ8()) << " != the recomputed " << waste
          << " (min(stride, cacheline/elem_bytes)-1)*256";
      return false;
    }
    if (conflict != static_cast<int64_t>(op.getBankConflictQ8())) {
      op.emitError("bcir-cache-contention: bank_conflict_q8 ")
          << static_cast<int64_t>(op.getBankConflictQ8()) << " != the recomputed " << conflict
          << " (gcd(stride, banks)-1)*256";
      return false;
    }
    if (cont != static_cast<int64_t>(op.getContentionQ8())) {
      op.emitError("bcir-cache-contention: contention_q8 ")
          << static_cast<int64_t>(op.getContentionQ8()) << " != line_waste_q8 + bank_conflict_q8 "
          << cont;
      return false;
    }
    if (cost != static_cast<int64_t>(op.getContentionCost())) {
      op.emitError("bcir-cache-contention: contention_cost ")
          << static_cast<int64_t>(op.getContentionCost()) << " != (contention_q8 * count) >> 8 "
          << cost;
      return false;
    }

    // --- annotate the recomputed CONTENTION signal (the cross-pass dual-rail record) ----
    // INFORMS-ONLY: these ride the CONTENTION axis and never gate legality (-bcir-verify
    // reads only R8/R9, never these). is_conflict_free / is_noop mirror ContentionPrediction.
    op->setAttr("kbcir.line_waste_q8", ab.getI64IntegerAttr(waste));
    op->setAttr("kbcir.bank_conflict_q8", ab.getI64IntegerAttr(conflict));
    op->setAttr("kbcir.contention_q8", ab.getI64IntegerAttr(cont));
    op->setAttr("kbcir.contention_cost", ab.getI64IntegerAttr(cost));
    op->setAttr("kbcir.contention_axis", ab.getStringAttr("CONTENTION")); // the cost axis it rides
    op->setAttr("kbcir.informs_only", ab.getBoolAttr(true)); // never a legality verdict
    op->setAttr("kbcir.conflict_free", ab.getBoolAttr(waste == 0 && conflict == 0));
    op->setAttr("kbcir.noop", ab.getBoolAttr(cost == 0));
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> createCacheContentionPass() {
  return std::make_unique<CacheContentionPass>();
}

} // namespace bcir
