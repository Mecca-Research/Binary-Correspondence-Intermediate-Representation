//===- BCIRComposePass.cpp - compositional cost over the func/if region tree -*- C++ -*-===//
//
// -bcir-compose: the law-rail port of bcir/kbcir/compose.plan_composite. It walks the
// kbcir.func / kbcir.call / kbcir.cond region tree (with bcir.claim ops as the leaves) and
// computes the compositional cost along the central equation's series-parallel grain:
//
//   Leaf (a region's direct bcir.claim ops)  : priced as one parallel block by the K_BCIR
//                                               cost model (realize.optimize / planChosen).
//   Seq  (the ops in a region, in series)     : the SUM of their costs.
//   Cond (kbcir.cond)                          : worst-case = PRED + max(then, else); the
//                                               expected = PRED + prob-weighted branch cost.
//   Call (kbcir.call)                          : the callee kbcir.func's cost (inlined, memoized).
//
// Annotates each kbcir.func with kbcir.compose_worst (a hard latency bound) + kbcir.compose_
// expected (the probability-weighted cost). Reproduces the oracle: a Leaf([vector_add]) func
// prices to exactly 7808, and a Seq/Cond/Call program to plan_composite's worst/expected.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRCostModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <utility>

using namespace mlir;

namespace bcir {
namespace {

// compose.PRED_COST: the fixed control-flow predicate overhead (score units).
static constexpr int64_t kPredCost = 8;

struct ComposePass : public PassWrapper<ComposePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComposePass)

  StringRef getArgument() const final { return "bcir-compose"; }
  StringRef getDescription() const final {
    return "Compositional cost over the kbcir.func/call/cond region tree (the law-rail "
           "compose.plan_composite): Seq = sum, Cond = worst-case max + probability-weighted "
           "expected, Call = inline, Leaf = optimize; annotates kbcir.compose_worst/expected.";
  }

  cm::Cap h;
  ArrayRef<int64_t> w;
  int64_t theta = 0;
  llvm::DenseMap<StringRef, ResourceOp> resByName;
  llvm::DenseMap<StringRef, Operation *> funcByName;
  llvm::DenseMap<StringRef, std::pair<int64_t, int64_t>> memo; // func -> (worst, expected)

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b);
    });
  }

  void runOnModule(Operation *root, Builder &b) {
    auto capOp = cm::firstCapability(root);
    if (!capOp)
      return;
    h = cm::readCap(capOp);
    w = cm::firstWeights(root);
    if (w.empty())
      return;
    theta = cm::firstThetaThermal(root);
    resByName = cm::resourcesByName(root);
    funcByName.clear();
    memo.clear();
    root->walk([&](KBCIRFuncOp f) { funcByName[f.getSymName()] = f.getOperation(); });
    for (auto &kv : funcByName) {
      std::pair<int64_t, int64_t> cost = funcCost(kv.first, 0);
      kv.second->setAttr("kbcir.compose_worst", b.getI64IntegerAttr(cost.first));
      kv.second->setAttr("kbcir.compose_expected", b.getI64IntegerAttr(cost.second));
    }
  }

  std::pair<int64_t, int64_t> funcCost(StringRef name, int depth) {
    auto it = memo.find(name);
    if (it != memo.end())
      return it->second;
    Operation *f = funcByName.lookup(name);
    if (!f || depth > 64)
      return {0, 0};
    memo[name] = {0, 0}; // break any cycle (R18 rejects recursion; be safe regardless)
    std::pair<int64_t, int64_t> cost = regionCost(f->getRegion(0).front(), depth);
    memo[name] = cost;
    return cost;
  }

  std::pair<int64_t, int64_t> regionCost(Block &block, int depth) {
    int64_t worst = 0, expected = 0;
    // Leaf: the region's direct bcir.claim ops, priced as one parallel block.
    SmallVector<ClaimOp> leaf;
    for (Operation &op : block)
      if (auto c = dyn_cast<ClaimOp>(&op))
        leaf.push_back(c);
    if (!leaf.empty()) {
      std::vector<cm::Column> cols = cm::fusedColumnsFromClaims(leaf, h, resByName);
      int64_t total = 0;
      cm::planChosen(cols, w, theta, total);
      worst += total;
      expected += total;
    }
    // Series composition with the nested control flow / calls.
    for (Operation &op : block) {
      if (auto cond = dyn_cast<KBCIRCondOp>(&op)) {
        std::pair<int64_t, int64_t> t = regionCost(cond.getThenRegion().front(), depth + 1);
        std::pair<int64_t, int64_t> e = regionCost(cond.getElseRegion().front(), depth + 1);
        int64_t p = std::max<int64_t>(
            0, std::min<int64_t>(1000, static_cast<int64_t>(cond.getProbThenMilli())));
        worst += kPredCost + std::max(t.first, e.first);
        expected += kPredCost + (p * t.second + (1000 - p) * e.second) / 1000;
      } else if (auto call = dyn_cast<KBCIRCallOp>(&op)) {
        std::pair<int64_t, int64_t> c = funcCost(call.getCallee(), depth + 1);
        worst += c.first;
        expected += c.second;
      }
    }
    return {worst, expected};
  }
};

} // namespace

std::unique_ptr<Pass> createComposePass() { return std::make_unique<ComposePass>(); }

} // namespace bcir
