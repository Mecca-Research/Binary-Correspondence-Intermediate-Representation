//===- BCIRLayoutPivotPass.cpp - the -bcir-layout-pivot pricing ------*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
// Recomputes the G3 SoA<->AoS layout-pivot PRICING DECISION for each
// `bcir.gem.layout_pivot` carrier op (the law-rail port of bcir/kbcir/layout.py::
// plan_layout + LayoutCertificate) and annotates the priced SoA/AoS cost + the chosen
// layout + the modeled gain on the op. Like -bcir-cache-contention this is a cost
// PRODUCER, not a lowering: the op is NOT erased -- the priced decision is annotated as
// kbcir.* attrs (a LayoutCertificate-equivalent), the cross-pass dual-rail record.
//
// THE PRICING THIS PASS REPRODUCES (mirrors layout.py exactly, verified by
// mlir/test/passes/layout_pivot.mlir + bcir/tests/test_layout_pivot_law_parity.py):
//   A multi-field (F-field) resource can be SoA (one contiguous array per field) or AoS
//   (records interleaved). A single-field SWEEP is UNIT under SoA / STRIDED(step=F) under
//   AoS; a whole-record access is the exact MIRROR (UNIT under AoS / STRIDED under SoA).
//   The cost of each shape under each layout is priced THROUGH THE SAME stride-penalty
//   MEMORY terms realize.candidates_for prices every claim by, taking the CHEAPEST
//   candidate's memory axis (NOT a bespoke layout discount):
//     UNIT (contiguous) min-mem = n*streams + ceil(n/vector_width)*streams*base_overhead*1
//                                 (vectorizes to the widest lane; pays base_overhead*1)
//     STRIDED (step F)  min-mem = n*streams + n*streams*base_overhead*min(F, cacheline/elem)
//                                 (pinned scalar width 1; pays the stride penalty)
//   The workload cost under a layout is the sum over the two shapes (each weighted by its
//   records). The MIN-cost layout wins; AoS is adopted ONLY on a STRICT win (ties keep the
//   declared default SoA). gain = the SoA (declared) cost minus the chosen-layout cost.
//
// THE HARD INVARIANT (layout changes ADDRESSES, never VALUES) is the ORACLE's, proved
// bit-exact by lower.layout_kernel; the law records only the PRICED choice. The priced
// layout/cost is INFORMS-ONLY: it informs the plan's memory cost but is NEVER read by
// -bcir-verify as a legality verdict (the same two-truth quarantine the CONTENTION signal
// preserves -- the memory-cost lever stays off the legality path).
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

struct LayoutPivotPass
    : public PassWrapper<LayoutPivotPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutPivotPass)

  StringRef getArgument() const final { return "bcir-layout-pivot"; }
  StringRef getDescription() const final {
    return "Recompute the G3 SoA<->AoS layout-pivot pricing for each gem.layout_pivot op "
           "(port of layout.py::plan_layout) through the same stride-penalty memory terms "
           "and annotate the priced cost + chosen layout + gain; INFORMS-ONLY (never a "
           "legality verdict).";
  }

  void runOnOperation() override {
    SmallVector<GEMLayoutPivotOp> ops;
    getOperation()->walk([&](GEMLayoutPivotOp p) { ops.push_back(p); });
    bool ok = true;
    for (GEMLayoutPivotOp p : ops)
      ok &= priceOne(p);
    if (!ok)
      signalPassFailure();
  }

  // Recompute the SoA-vs-AoS pricing from the op's access workload + frozen geometry, cross-
  // check the declared decision, and annotate it. Returns false on a parity violation. (The
  // op is NOT erased -- this is a cost producer, not a lowering.)
  bool priceOne(GEMLayoutPivotOp op) {
    MLIRContext *ctx = op.getContext();
    Builder ab(ctx);

    int64_t fields = std::max<int64_t>(1, static_cast<int64_t>(op.getFields()));
    int64_t sfr = std::max<int64_t>(0, static_cast<int64_t>(op.getSingleFieldRecords()));
    int64_t wrr = std::max<int64_t>(0, static_cast<int64_t>(op.getWholeRecordRecords()));
    int64_t cl = std::max<int64_t>(1, static_cast<int64_t>(op.getCacheline()));
    int64_t eb = std::max<int64_t>(1, static_cast<int64_t>(op.getElemBytes()));
    int64_t bo = std::max<int64_t>(1, static_cast<int64_t>(op.getBaseOverhead()));
    int64_t streams = std::max<int64_t>(1, static_cast<int64_t>(op.getStreams()));
    int64_t vw = std::max<int64_t>(1, static_cast<int64_t>(op.getVectorWidth()));
    int64_t gp = std::max<int64_t>(1, static_cast<int64_t>(op.getGatherPenalty()));

    // --- price each access shape under each layout through the stride-penalty memory terms ---
    // STRIDED penalty: min(F, cacheline/elem_bytes), clamped to >= 2 like the oracle's stride_k_under.
    int64_t sp = std::min<int64_t>(std::max<int64_t>(2, fields), std::max<int64_t>(1, cl / eb));
    auto unitMem = [&](int64_t n, int64_t &out) -> bool {
      if (n <= 0)
        return out = 0, true;
      int64_t ceil = n / vw + (n % vw != 0);
      int64_t base, overhead;
      return checkedMulNonnegative(n, streams, base) &&
             checkedMul3Nonnegative(ceil, streams, bo, overhead) &&
             checkedAddNonnegative(base, overhead, out);
    };
    auto stridedMem = [&](int64_t n, int64_t &out) -> bool {
      if (n <= 0)
        return out = 0, true;
      // the cheapest STRIDED candidate = min(strided penalty, gather penalty): a target whose gather is
      // cheaper than a large stride (e.g. PTX gp=16) prices the gather instead (mirrors candidates_for).
      int64_t base, stridedExtra, gatherExtra, strided, gather;
      if (!checkedMulNonnegative(n, streams, base) ||
          !checkedMul3Nonnegative(base, bo, sp, stridedExtra) ||
          !checkedMul3Nonnegative(base, bo, gp, gatherExtra) ||
          !checkedAddNonnegative(base, stridedExtra, strided) ||
          !checkedAddNonnegative(base, gatherExtra, gather))
        return false;
      out = std::min(strided, gather);
      return true;
    };
    // single-field: UNIT under SoA / STRIDED under AoS; whole-record: STRIDED under SoA / UNIT under AoS.
    int64_t soa, aos, sfUnit, sfStrided, wrUnit, wrStrided;
    if (!unitMem(sfr, sfUnit) || !stridedMem(sfr, sfStrided) ||
        !unitMem(wrr, wrUnit) || !stridedMem(wrr, wrStrided)) {
      op.emitError("bcir-layout-pivot: derived memory cost exceeds signed 64-bit range");
      return false;
    }
    if (fields <= 1) {
      // No SoA/AoS distinction (a single-field resource): both costs equal (a clean no-op).
      if (!checkedAddNonnegative(sfUnit, wrUnit, soa)) {
        op.emitError("bcir-layout-pivot: aggregate memory cost exceeds signed 64-bit range");
        return false;
      }
      aos = soa;
    } else {
      if (!checkedAddNonnegative(sfUnit, wrStrided, soa) ||
          !checkedAddNonnegative(sfStrided, wrUnit, aos)) {
        op.emitError("bcir-layout-pivot: aggregate memory cost exceeds signed 64-bit range");
        return false;
      }
    }
    // min,+ selection: AoS adopted ONLY on a STRICT win; ties keep the declared default SoA.
    StringRef layout = (aos < soa) ? "aos" : "soa";
    int64_t chosen = (layout == "aos") ? aos : soa;
    int64_t gain = soa - chosen;                          // declared (SoA) cost minus chosen cost (>= 0)

    // --- cross-check the declared decision (the dual-rail parity gate) -----------------
    if (soa != static_cast<int64_t>(op.getSoaCost())) {
      op.emitError("bcir-layout-pivot: soa_cost ")
          << static_cast<int64_t>(op.getSoaCost()) << " != the recomputed priced SoA cost " << soa;
      return false;
    }
    if (aos != static_cast<int64_t>(op.getAosCost())) {
      op.emitError("bcir-layout-pivot: aos_cost ")
          << static_cast<int64_t>(op.getAosCost()) << " != the recomputed priced AoS cost " << aos;
      return false;
    }
    if (layout != op.getLayout()) {
      op.emitError("bcir-layout-pivot: layout '")
          << op.getLayout() << "' != the min-cost choice '" << layout
          << "' (ties keep the default soa; soa " << soa << " vs aos " << aos << ")";
      return false;
    }
    if (gain != static_cast<int64_t>(op.getGain())) {
      op.emitError("bcir-layout-pivot: gain ")
          << static_cast<int64_t>(op.getGain()) << " != soa_cost - chosen_cost " << gain;
      return false;
    }

    // --- annotate the priced layout decision (the LayoutCertificate-equivalent record) --
    // INFORMS-ONLY: the priced layout/cost informs the plan's memory cost but never gates
    // legality (-bcir-verify never reads these). is_soa/is_aos/is_noop mirror LayoutCertificate.
    op->setAttr("kbcir.layout", ab.getStringAttr(layout));
    op->setAttr("kbcir.soa_cost", ab.getI64IntegerAttr(soa));
    op->setAttr("kbcir.aos_cost", ab.getI64IntegerAttr(aos));
    op->setAttr("kbcir.chosen_cost", ab.getI64IntegerAttr(chosen));
    op->setAttr("kbcir.gain", ab.getI64IntegerAttr(gain));
    op->setAttr("kbcir.declared_layout", ab.getStringAttr("soa"));  // the baseline we pivot from
    op->setAttr("kbcir.informs_only", ab.getBoolAttr(true));        // never a legality verdict
    op->setAttr("kbcir.noop", ab.getBoolAttr(layout == "soa" && gain == 0));
    return true;
  }
};

}  // namespace

std::unique_ptr<Pass> createLayoutPivotPass() {
  return std::make_unique<LayoutPivotPass>();
}

}  // namespace bcir
