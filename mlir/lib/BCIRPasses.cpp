//===- BCIRPasses.cpp - BCIR compiler passes ------------------------------===//
//
// Phase 6: turn the MLIR rail into a real compiler. The Python oracle stays the
// conformance reference (docs/PARITY.md); these passes are the MLIR-native
// implementations of the same laws.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace bcir {
namespace {

//===----------------------------------------------------------------------===//
// -bcir-verify : semantic laws R1 / R2 / R4 / R6 (mirrors bcir/verify).
//===----------------------------------------------------------------------===//

// Which lanes are legal for each access-pattern shape (LangRef R6).
static bool laneLegalForStride(Lane lane, StrideClass sc) {
  switch (sc) {
  case StrideClass::Scalar:
    return lane == Lane::U || lane == Lane::H;
  case StrideClass::Unit:
    return lane == Lane::U;
  case StrideClass::Strided:
    return lane == Lane::U || lane == Lane::GGG;
  case StrideClass::Cacheline:
    return lane == Lane::UX || lane == Lane::GGG;
  case StrideClass::Tile:
    return lane == Lane::T;
  case StrideClass::Random:
    return lane == Lane::GGG || lane == Lane::A;
  }
  return false;
}

struct VerifyPass : public PassWrapper<VerifyPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyPass)

  StringRef getArgument() const final { return "bcir-verify"; }
  StringRef getDescription() const final {
    return "Verify BCIR semantic laws R1 (RID uniqueness), R2 (resource "
           "resolution), R4 (phase DAG acyclicity), R6 (lane/stride legality).";
  }

  void runOnOperation() override {
    Operation *root = getOperation();
    bool ok = true;

    // R1: registry uniqueness -- every RID unique.
    llvm::DenseMap<uint32_t, ResourceOp> rids;
    llvm::DenseSet<StringRef> resourceNames;
    root->walk([&](ResourceOp r) {
      resourceNames.insert(r.getSymName());
      uint32_t rid = r.getRid();
      if (rids.count(rid)) {
        r.emitError("R1: duplicate RID ") << rid;
        ok = false;
      } else {
        rids[rid] = r;
      }
    });

    // R2: registry resolution -- claim reads/writes resolve to declared resources.
    auto checkRefs = [&](ClaimOp c, ArrayAttr refs, StringRef which) {
      for (Attribute a : refs) {
        auto ref = dyn_cast<FlatSymbolRefAttr>(a);
        if (ref && !resourceNames.count(ref.getValue())) {
          c.emitError("R2: claim ")
              << c.getSymName() << " " << which << " undeclared resource @"
              << ref.getValue();
          ok = false;
        }
      }
    };
    root->walk([&](ClaimOp c) {
      checkRefs(c, c.getReads(), "reads");
      checkRefs(c, c.getWrites(), "writes");
    });

    // R4: phase DAG legality -- dependencies form an acyclic graph.
    llvm::DenseMap<StringRef, SmallVector<StringRef, 2>> deps;
    llvm::DenseMap<StringRef, PhaseOp> phaseOps;
    SmallVector<StringRef> phases;
    root->walk([&](PhaseOp p) {
      phases.push_back(p.getSymName());
      phaseOps[p.getSymName()] = p;
      for (Attribute a : p.getDeps())
        if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
          deps[p.getSymName()].push_back(ref.getValue());
    });
    // 0 = unvisited, 1 = on stack, 2 = done.
    llvm::DenseMap<StringRef, int> color;
    std::function<bool(StringRef)> hasCycle = [&](StringRef n) -> bool {
      color[n] = 1;
      for (StringRef d : deps.lookup(n)) {
        int c = color.lookup(d);
        if (c == 1)
          return true;
        if (c == 0 && hasCycle(d))
          return true;
      }
      color[n] = 2;
      return false;
    };
    for (StringRef p : phases) {
      if (color.lookup(p) == 0 && hasCycle(p)) {
        phaseOps[p].emitError("R4: phase dependency cycle through @") << p;
        ok = false;
        break;
      }
    }

    // R6: lane legality -- lane matches the declared access pattern.
    root->walk([&](ClaimOp c) {
      if (!laneLegalForStride(c.getLane(), c.getStrideClass())) {
        c.emitError("R6: claim ")
            << c.getSymName() << " lane illegal for its stride class";
        ok = false;
      }
    });

    if (!ok)
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// -bcir-promote-lanes : the opt-law GGG -> UX promotion (a rewrite).
//===----------------------------------------------------------------------===//

// A claim declared GGG/Random carrying a `bcir.bucketable` proof attribute is
// promoted to UX/Cacheline (LangRef Sec. 11; mirrors bcir.opt.rewrite_rule
// @ggg_to_ux). Legal only with the proof; never invents locality.
struct PromoteGGGtoUX : public OpRewritePattern<ClaimOp> {
  using OpRewritePattern<ClaimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ClaimOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLane() != Lane::GGG || op.getStrideClass() != StrideClass::Random)
      return failure();
    auto proof = op->getAttrOfType<BoolAttr>("bcir.bucketable");
    if (!proof || !proof.getValue())
      return failure();
    rewriter.modifyOpInPlace(op, [&] {
      op.setLaneAttr(LaneAttr::get(op.getContext(), Lane::UX));
      op.setStrideClassAttr(
          StrideClassAttr::get(op.getContext(), StrideClass::Cacheline));
    });
    return success();
  }
};

struct PromoteLanesPass
    : public PassWrapper<PromoteLanesPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteLanesPass)

  StringRef getArgument() const final { return "bcir-promote-lanes"; }
  StringRef getDescription() const final {
    return "Apply the BCIR lane-promotion opt-law (GGG -> UX with proof).";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PromoteGGGtoUX>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// -convert-bcir-to-llvm : BCIR compute/barrier -> LLVM dialect.
//===----------------------------------------------------------------------===//

struct ComputeOpLowering : public OpConversionPattern<ComputeOp> {
  using OpConversionPattern<ComputeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "only binary compute is lowered");
    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    StringRef kind = op.getKind();
    Value res;
    if (kind == "fadd")
      res = rewriter.create<LLVM::FAddOp>(op.getLoc(), lhs, rhs);
    else if (kind == "fsub")
      res = rewriter.create<LLVM::FSubOp>(op.getLoc(), lhs, rhs);
    else if (kind == "fmul")
      res = rewriter.create<LLVM::FMulOp>(op.getLoc(), lhs, rhs);
    else
      return rewriter.notifyMatchFailure(op, "unsupported compute kind");
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct BarrierOpLowering : public OpConversionPattern<BarrierOp> {
  using OpConversionPattern<BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Phase-8 memory model: the BCIR barrier ordering maps to the LLVM fence
    // ordering (default seq_cst). Fences require >= acquire, so unordered/
    // monotonic fall back to seq_cst.
    LLVM::AtomicOrdering ord = LLVM::AtomicOrdering::seq_cst;
    if (auto o = op.getOrdering()) {
      switch (*o) {
      case MemOrdering::Acquire: ord = LLVM::AtomicOrdering::acquire; break;
      case MemOrdering::Release: ord = LLVM::AtomicOrdering::release; break;
      case MemOrdering::AcqRel:  ord = LLVM::AtomicOrdering::acq_rel; break;
      default:                   ord = LLVM::AtomicOrdering::seq_cst; break;
      }
    }
    rewriter.replaceOpWithNewOp<LLVM::FenceOp>(op, ord, StringRef());
    return success();
  }
};

struct ConvertToLLVMPass
    : public PassWrapper<ConvertToLLVMPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToLLVMPass)

  StringRef getArgument() const final { return "convert-bcir-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower BCIR compute/barrier to the LLVM dialect (partial).";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<ComputeOp, BarrierOp>();

    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<ComputeOpLowering, BarrierOpLowering>(typeConverter,
                                                       &getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createVerifyPass() {
  return std::make_unique<VerifyPass>();
}
std::unique_ptr<Pass> createPromoteLanesPass() {
  return std::make_unique<PromoteLanesPass>();
}
std::unique_ptr<Pass> createConvertToLLVMPass() {
  return std::make_unique<ConvertToLLVMPass>();
}

void registerBCIRPasses() {
  PassRegistration<VerifyPass>();
  PassRegistration<PromoteLanesPass>();
  PassRegistration<ConvertToLLVMPass>();
}

}  // namespace bcir
