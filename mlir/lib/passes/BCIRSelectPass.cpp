//===- BCIRSelectPass.cpp - the -bcir-select-realization min-plus selection -*- C++ -*-===//
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
struct SelectRealizationPass : public PassWrapper<SelectRealizationPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectRealizationPass)

  StringRef getArgument() const final { return "bcir-select-realization"; }
  StringRef getDescription() const final {
    return "K_BCIR min-plus realization selection over the candidate DAG "
           "(reproduces the oracle score; annotates kbcir.computed_*).";
  }

  void runOnOperation() override {
    Operation *root = getOperation();
    Builder b(&getContext());
    bool ok = true;

    llvm::DenseMap<StringRef, KBCIRPathOp> pathByName;
    llvm::DenseMap<StringRef, KBCIRBudgetOp> budgetByName;
    llvm::DenseMap<int, KBCIRPolicyOp> policyByMode; // PolicyMode -> first policy
    root->walk([&](KBCIRPathOp p) { pathByName[p.getSymName()] = p; });
    root->walk([&](KBCIRBudgetOp bud) { budgetByName[bud.getSymName()] = bud; });
    root->walk(
        [&](KBCIRPolicyOp p) { policyByMode.try_emplace(static_cast<int>(p.getMode()), p); });

    root->walk([&](KBCIRSelectOp s) {
      auto polIt = policyByMode.find(static_cast<int>(s.getPolicy()));
      if (polIt == policyByMode.end()) {
        s.emitError("bcir-select-realization: no kbcir.policy declares the "
                    "selection's policy mode");
        ok = false;
        return;
      }
      ArrayRef<int64_t> w = polIt->second.getWeights();
      if (w.size() != 12 ||
          std::any_of(w.begin(), w.end(), [](int64_t value) { return value < 0; })) {
        s.emitError("bcir-select-realization: policy must contain exactly 12 non-negative weights");
        ok = false;
        return;
      }

      // Optional budget (RCSP rail): a candidate is feasible iff every named
      // cost dim is within its cap.
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

      // argmin over the feasible candidates (deterministic: first wins on tie).
      StringRef bestName;
      int64_t bestScore = 0;
      bool have = false;
      for (Attribute a : s.getFrom()) {
        auto ref = dyn_cast<FlatSymbolRefAttr>(a);
        if (!ref)
          continue;
        auto p = pathByName.lookup(ref.getValue());
        if (!p || !feasible(p))
          continue;
        int64_t sc = scalarize(p.getCost(), w);
        if (!have || sc < bestScore) {
          have = true;
          bestScore = sc;
          bestName = ref.getValue();
        }
      }
      if (!have) {
        s.emitError("bcir-select-realization: no feasible candidate for claim @") << s.getClaim();
        ok = false;
        return;
      }

      // Annotate the computed plan, then cross-check the declared selection.
      s->setAttr("kbcir.computed_selected", FlatSymbolRefAttr::get(&getContext(), bestName));
      s->setAttr("kbcir.computed_score", b.getI64IntegerAttr(bestScore));
      if (bestName != s.getSelected()) {
        s.emitError("bcir-select-realization: computed argmin @")
            << bestName << " != declared selected @" << s.getSelected();
        ok = false;
      }
      if (bestScore != static_cast<int64_t>(s.getScore())) {
        s.emitError("bcir-select-realization: computed score ")
            << bestScore << " != declared score " << static_cast<int64_t>(s.getScore());
        ok = false;
      }
    });
    if (!ok)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createSelectRealizationPass() {
  return std::make_unique<SelectRealizationPass>();
}

} // namespace bcir
