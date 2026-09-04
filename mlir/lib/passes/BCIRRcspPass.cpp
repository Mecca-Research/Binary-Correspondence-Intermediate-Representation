//===- BCIRRcspPass.cpp - the -bcir-rcsp constrained selection + Pareto -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIR/BCIRPasses.h"
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
struct RcspPass : public PassWrapper<RcspPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RcspPass)

  StringRef getArgument() const final { return "bcir-rcsp"; }
  StringRef getDescription() const final {
    return "K_BCIR constrained selection (RCSP): budget-feasible label-DP argmin + "
           "the Pareto front over (score, thermal, power); MLIR-native port of "
           "bcir/kbcir/rcsp (annotates kbcir.computed_* + kbcir.pareto_size).";
  }

  // A Pareto label: the scalarized score plus the two tracked resource dims.
  using Label = std::array<int64_t, 3>;
  static bool dominates(const Label &a, const Label &b) {
    return a[0] <= b[0] && a[1] <= b[1] && a[2] <= b[2]; // weak (<=) on every axis
  }

  void runOnOperation() override {
    // S0-5 (module scope): the constrained selection prices the paths, budgets and
    // policies of ITS `bcir.module`, like -bcir-select-realization. See BCIRPassSupport.h.
    Builder b(&getContext());
    bool ok = true;
    forEachScope(getOperation(), [&](Operation *scope) { ok &= rcspScope(scope, b); });
    if (!ok)
      signalPassFailure();
  }

  bool rcspScope(Operation *root, Builder &b) {
    bool ok = true;

    llvm::DenseMap<StringRef, KBCIRPathOp> pathByName;
    llvm::DenseMap<StringRef, KBCIRBudgetOp> budgetByName;
    llvm::DenseMap<int, KBCIRPolicyOp> policyByMode;
    walkScope(root, [&](KBCIRPathOp p) { pathByName[p.getSymName()] = p; });
    walkScope(root, [&](KBCIRBudgetOp bd) { budgetByName[bd.getSymName()] = bd; });
    walkScope(root,
              [&](KBCIRPolicyOp p) { policyByMode.try_emplace(static_cast<int>(p.getMode()), p); });

    walkScope(root, [&](KBCIRSelectOp s) {
      auto polIt = policyByMode.find(static_cast<int>(s.getPolicy()));
      if (polIt == policyByMode.end()) {
        s.emitError("bcir-rcsp: no kbcir.policy declares the selection's policy mode");
        ok = false;
        return;
      }
      ArrayRef<int64_t> w = polIt->second.getWeights();

      KBCIRBudgetOp budget;
      if (auto bref = s.getBudgetAttr())
        budget = budgetByName.lookup(bref.getValue());
      auto feasible = [&](KBCIRPathOp p) {
        if (!budget)
          return true;
        ArrayAttr dims = budget.getDims();
        ArrayRef<int64_t> caps = budget.getCaps();
        for (size_t i = 0; i < dims.size() && i < caps.size(); ++i)
          if (auto name = dyn_cast<StringAttr>(dims[i]))
            if (auto v = costDim(p.getCost(), name.getValue()))
              if (*v > caps[i])
                return false;
        return true;
      };

      // The candidate paths in declared (`from`) order -- the deterministic tie
      // order (first label wins, as in the oracle's stable label DP).
      SmallVector<KBCIRPathOp> cands;
      for (Attribute a : s.getFrom())
        if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
          if (auto p = pathByName.lookup(ref.getValue()))
            cands.push_back(p);

      auto labelOf = [&](KBCIRPathOp p) -> Label {
        return {scalarize(p.getCost(), w), costDim(p.getCost(), "thermal").value_or(0),
                costDim(p.getCost(), "power").value_or(0)};
      };

      // (1) budget-feasible min-plus argmin (the constrained optimum).
      StringRef bestName;
      int64_t bestScore = 0;
      bool have = false;
      for (KBCIRPathOp p : cands) {
        if (!feasible(p))
          continue;
        int64_t sc = scalarize(p.getCost(), w);
        if (!have || sc < bestScore) {
          have = true;
          bestScore = sc;
          bestName = p.getSymName();
        }
      }
      if (!have) {
        s.emitError("bcir-rcsp: no feasible candidate for claim @") << s.getClaim();
        ok = false;
        return;
      }

      // (2) the Pareto front over (score, thermal, power) -- non-dominated labels,
      // inserted in score order with dominance pruning (mirrors rcsp.pareto_plans /
      // _insert). Unconstrained: the front is a property of the candidate set.
      SmallVector<KBCIRPathOp> order(cands.begin(), cands.end());
      llvm::stable_sort(order,
                        [&](KBCIRPathOp x, KBCIRPathOp y) { return labelOf(x) < labelOf(y); });
      SmallVector<Label> front;
      for (KBCIRPathOp p : order) {
        Label lp = labelOf(p);
        bool dominated = false;
        for (const Label &f : front)
          if (dominates(f, lp) && f != lp) { // a strictly-better incumbent excludes p
            dominated = true;
            break;
          }
        if (dominated)
          continue;
        front.erase(std::remove_if(front.begin(), front.end(),
                                   [&](const Label &f) { return dominates(lp, f) && lp != f; }),
                    front.end());
        front.push_back(lp);
      }

      s->setAttr("kbcir.computed_selected", FlatSymbolRefAttr::get(&getContext(), bestName));
      s->setAttr("kbcir.computed_score", b.getI64IntegerAttr(bestScore));
      s->setAttr("kbcir.pareto_size", b.getI64IntegerAttr(static_cast<int64_t>(front.size())));
      if (bestName != s.getSelected()) {
        s.emitError("bcir-rcsp: computed argmin @")
            << bestName << " != declared selected @" << s.getSelected();
        ok = false;
      }
      if (bestScore != static_cast<int64_t>(s.getScore())) {
        s.emitError("bcir-rcsp: computed score ")
            << bestScore << " != declared score " << static_cast<int64_t>(s.getScore());
        ok = false;
      }
    });
    return ok;
  }
};

} // namespace

std::unique_ptr<Pass> createRcspPass() {
  return std::make_unique<RcspPass>();
}

} // namespace bcir
