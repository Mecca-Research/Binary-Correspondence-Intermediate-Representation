//===- BCIRGEMPasses.cpp - the GEM pipeline: classify/batch/schedule/lower -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRPassSupport.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <array>
#include <functional>
#include <optional>

using namespace mlir;

namespace bcir {
namespace {
struct ClassifyLanesPass
    : public PassWrapper<ClassifyLanesPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClassifyLanesPass)

  StringRef getArgument() const final { return "bcir-classify-lanes"; }
  StringRef getDescription() const final {
    return "Classify each claim's access pattern into its canonical lane "
           "(annotates kbcir.classified_lane; mirrors the oracle classifier).";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](ClaimOp c) {
      Lane lane = canonicalLane(c.getStrideClass());
      c->setAttr("kbcir.classified_lane", b.getStringAttr(laneSpelling(lane)));
    });
  }
};

struct BatchPass : public PassWrapper<BatchPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchPass)

  StringRef getArgument() const final { return "bcir-batch"; }
  StringRef getDescription() const final {
    return "Form GEM batches from same-phase fusion-compatible claims "
           "(annotates kbcir.batch).";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    // Collect claims per phase, ordered by claim id (deterministic).
    llvm::DenseMap<StringRef, SmallVector<ClaimOp>> byPhase;
    getOperation()->walk([&](ClaimOp c) { byPhase[c.getPhase()].push_back(c); });
    int64_t nextBatch = 0;
    for (auto &kv : byPhase) {
      SmallVector<ClaimOp> &claims = kv.second;
      llvm::sort(claims, [](ClaimOp a, ClaimOp b) {
        return a.getClaimId() < b.getClaimId();
      });
      SmallVector<ClaimOp> open;  // claims already placed in the current batch
      int64_t batch = -1;
      for (ClaimOp c : claims) {
        bool fits = batch >= 0;
        for (ClaimOp o : open)
          if (o.getLane() != c.getLane() || claimsConflict(o, c)) {
            fits = false;
            break;
          }
        if (!fits) {
          batch = nextBatch++;
          open.clear();
        }
        c->setAttr("kbcir.batch", b.getI64IntegerAttr(batch));
        open.push_back(c);
      }
    }
  }
};

struct SchedulePass : public PassWrapper<SchedulePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SchedulePass)

  StringRef getArgument() const final { return "bcir-schedule"; }
  StringRef getDescription() const final {
    return "Phase-sliced deterministic scheduling (annotates kbcir.exec_order).";
  }

  void runOnOperation() override {
    Operation *root = getOperation();
    Builder b(&getContext());
    llvm::DenseMap<StringRef, int32_t> phaseId;
    root->walk([&](PhaseOp p) { phaseId[p.getSymName()] = p.getId(); });

    SmallVector<ClaimOp> claims;
    root->walk([&](ClaimOp c) { claims.push_back(c); });
    llvm::sort(claims, [&](ClaimOp a, ClaimOp b) {
      int32_t pa = phaseId.lookup(a.getPhase()), pb = phaseId.lookup(b.getPhase());
      if (pa != pb)
        return pa < pb;
      return a.getClaimId() < b.getClaimId();
    });
    int64_t order = 0;
    for (ClaimOp c : claims)
      c->setAttr("kbcir.exec_order", b.getI64IntegerAttr(order++));
  }
};

struct LowerToLLVMPass : public PassWrapper<LowerToLLVMPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLLVMPass)

  StringRef getArgument() const final { return "bcir-lower-to-llvm"; }
  StringRef getDescription() const final {
    return "Check the GEM StreamPack lowering contract (lane/resource "
           "preservation, R12) and mark segments lowered.";
  }

  void runOnOperation() override {
    Operation *root = getOperation();
    Builder b(&getContext());
    bool ok = true;
    llvm::DenseMap<StringRef, ClaimOp> claimByName;
    root->walk([&](ClaimOp c) { claimByName[c.getSymName()] = c; });

    root->walk([&](GEMLaneSegmentOp seg) {
      auto c = claimByName.lookup(seg.getClaim());
      if (!c) {
        seg.emitError("bcir-lower-to-llvm: segment references unknown claim @")
            << seg.getClaim();
        ok = false;
        return;
      }
      if (seg.getLane() != c.getLane()) {
        seg.emitError("bcir-lower-to-llvm: segment lane not preserved (R12): ")
            << laneSpelling(seg.getLane()) << " != claim lane "
            << laneSpelling(c.getLane());
        ok = false;
      }
      if (seg.getReads() != c.getReads() || seg.getWrites() != c.getWrites()) {
        seg.emitError("bcir-lower-to-llvm: segment resource set not preserved "
                      "(R12) for claim @")
            << seg.getClaim();
        ok = false;
      }
      // R14 (CIM/PIM dispatch legality): a segment dispatched to processing-in-memory
      // must be a reduction -- PIM does element-local reduce work, not general SIMD.
      // Mirrors gem.cim in the oracle (only reduce.* claims are offloaded).
      if (seg.getDispatch() == "pim" && !seg.getOpcode().starts_with("reduce.")) {
        seg.emitError("R14: pim dispatch illegal for non-reduction op '")
            << seg.getOpcode() << "' (claim @" << seg.getClaim() << ")";
        ok = false;
      }
      // R15 (DVFS clock legality): the clock must be a legal step in [64, 512]
      // (0.25x..2x Q8), and a pim (memory-bound) segment must not overclock -- more
      // core frequency does not help bandwidth-bound work. Mirrors gem.dvfs.
      int64_t clk = seg.getClockQ8();
      if (clk < 64 || clk > 512) {
        seg.emitError("R15: clock_q8 ") << clk << " out of legal range [64, 512]";
        ok = false;
      } else if (seg.getDispatch() == "pim" && clk > 256) {
        seg.emitError("R15: pim (memory-bound) segment must not overclock (clock_q8 ")
            << clk << " > 256)";
        ok = false;
      }
      seg->setAttr("kbcir.lowered", b.getBoolAttr(true));
    });

    // R16 (allocator placement legality): an on-chip placement must fit. A resource
    // placed in L1 must be <= 64 KiB, in L2 <= 4 MiB (static caps; size =
    // product(shape) * 4 B). Mirrors kbcir.allocator's capacity gate -- the planner
    // may not promote a tensor into an SRAM tier it cannot fit. L3/DRAM/HBM: no cap.
    root->walk([&](ResourceOp r) {
      auto pl = r.getPlacement();
      if (!pl)
        return;
      ArrayRef<int64_t> shape = r.getShape();
      if (shape.empty())
        return;  // dynamic extent: not statically checkable
      int64_t bytes = 4;
      for (int64_t d : shape)
        bytes = saturatingMulNonnegative(bytes, d > 0 ? d : 1);
      int64_t cap = (*pl == MemTier::L1) ? (64 * 1024)
                  : (*pl == MemTier::L2) ? (4 * 1024 * 1024) : 0;
      if (cap && bytes > cap) {
        r.emitError("R16: placement ") << stringifyMemTier(*pl) << " does not fit @"
            << r.getSymName() << " (" << bytes << " B > " << cap << " B)";
        ok = false;
      }
    });
    if (!ok)
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createClassifyLanesPass() {
  return std::make_unique<ClassifyLanesPass>();
}
std::unique_ptr<Pass> createBatchPass() { return std::make_unique<BatchPass>(); }
std::unique_ptr<Pass> createSchedulePass() {
  return std::make_unique<SchedulePass>();
}
std::unique_ptr<Pass> createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}

}  // namespace bcir
