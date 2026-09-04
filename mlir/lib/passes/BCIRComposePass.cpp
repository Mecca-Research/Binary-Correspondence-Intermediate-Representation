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
//   Call (kbcir.call)                          : INTER-PROCEDURAL SUMMARY -- a kbcir.func is
//                                               planned ONCE over its formals (memoized); a
//                                               call whose actuals are cost-compatible with
//                                               the formals (same domain / element-count /
//                                               access -- compose._cost_key) reuses that
//                                               summary, else the body is re-priced with the
//                                               actuals substituted (the inline fallback).
//
// Annotates each kbcir.func with kbcir.compose_worst (a hard latency bound), kbcir.compose_
// expected (the probability-weighted cost), and kbcir.compose_reused (leaf plans saved by
// summary reuse). Reproduces the oracle: a Leaf([vector_add]) func prices to exactly 7808,
// and a program reusing a func for a compatible call + re-pricing an HBM call to 10624 / 1.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIR/BCIRPasses.h"
#include "BCIRCostModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <utility>

using namespace mlir;

namespace bcir {
namespace {

// compose.PRED_COST: the fixed control-flow predicate overhead (score units).
static constexpr int64_t kPredCost = 8;

// A region's compositional cost (compose.CompositeResult): a hard worst-case bound, the
// probability-weighted expected cost, the planned-leaf count, the leaves saved by reuse, and
// (under a budget) whether every parallel block fit the caps.
struct CC {
  int64_t worst = 0, expected = 0, leaves = 0, reused = 0;
  bool feasible = true;
  bool dynamic = false; // any leaf is a dynamic-shape claim -> the plan is a worst-case bound
};

// A region's read/write footprint (compose.Effect) -- resource symbols, folded through calls.
struct Effect {
  llvm::DenseSet<StringRef> reads, writes;
};

// compose.Effect.conflicts: a RAW/WAR/WAW between two footprints (a write that aliases the
// other's read or write). Disjoint footprints commute (may be reordered / overlapped).
static bool effectsConflict(const Effect &a, const Effect &b) {
  for (StringRef wsym : a.writes)
    if (b.reads.count(wsym) || b.writes.count(wsym))
      return true;
  for (StringRef wsym : b.writes)
    if (a.reads.count(wsym))
      return true;
  return false;
}

// compose._cost_key equality: a summary is reusable for an actual iff it matches the formal
// on (domain, element count = product(shape), access).
static bool costKeyEq(ResourceOp a, ResourceOp b) {
  if (a.getDomainKind() != b.getDomainKind())
    return false;
  int64_t ca = 1, cb = 1;
  for (int64_t d : a.getShape())
    if (!checkedMulNonnegative(ca, d, ca))
      return false;
  for (int64_t d : b.getShape())
    if (!checkedMulNonnegative(cb, d, cb))
      return false;
  if (ca != cb)
    return false;
  bool ha = a.getAccess() && *a.getAccess() == Access::HAM;
  bool hb = b.getAccess() && *b.getAccess() == Access::HAM;
  return ha == hb;
}

struct ComposePass : public PassWrapper<ComposePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComposePass)

  StringRef getArgument() const final { return "bcir-compose"; }
  StringRef getDescription() const final {
    return "Compositional cost over the kbcir.func/call/cond region tree (the law-rail "
           "compose.plan_composite): Seq = sum, Cond = worst-case max + probability-weighted "
           "expected, Call = inter-procedural summary (plan once, reuse for cost-compatible "
           "calls, else re-price); annotates kbcir.compose_worst/expected/reused/feasible/"
           "dynamic, the kbcir.effect_reads/writes footprint, and kbcir.commutes_with_prev.";
  }

  cm::Cap h;
  ArrayRef<int64_t> w;
  int64_t theta = 0;
  llvm::DenseMap<StringRef, ResourceOp> resByName;
  llvm::DenseMap<StringRef, Operation *> funcByName;
  llvm::DenseMap<StringRef, CC> memo; // func -> its summary (planned once over its formals)
  llvm::DenseSet<StringRef> activeSummaries;
  llvm::DenseMap<StringRef, StringRef> emptySubst;
  SmallVector<int> budgetDims; // the first kbcir.budget's tracked cost-dim indices
  SmallVector<int64_t> budgetCaps;
  bool hasBudget = false; // a budget op present -> price Leaves constrained (RCSP)
  bool invalidNesting = false;

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
    if (w.size() != 12 || llvm::any_of(w, [](int64_t value) { return value < 0; }))
      return;
    theta = cm::firstThetaThermal(root);
    resByName = cm::resourcesByName(root);
    funcByName.clear();
    memo.clear();
    activeSummaries.clear();
    invalidNesting = false;
    // The first kbcir.budget (if any) makes each Leaf priced by the constrained label DP, so
    // the compositional plan respects min M s.t. R <= B (the central K_BCIR equation).
    budgetDims.clear();
    budgetCaps.clear();
    hasBudget = false;
    KBCIRBudgetOp budget;
    root->walk([&](KBCIRBudgetOp bd) {
      if (!budget)
        budget = bd;
    });
    if (budget) {
      ArrayAttr dims = budget.getDims();
      ArrayRef<int64_t> capArr = budget.getCaps();
      for (size_t i = 0; i < dims.size() && i < capArr.size(); ++i)
        if (auto nm = dyn_cast<StringAttr>(dims[i])) {
          int d = cm::dimIndex(nm.getValue());
          if (d >= 0) {
            budgetDims.push_back(d);
            budgetCaps.push_back(capArr[i]);
          }
        }
      hasBudget = !budgetDims.empty();
    }
    root->walk([&](KBCIRFuncOp f) { funcByName[f.getSymName()] = f.getOperation(); });
    for (auto &kv : funcByName) {
      Operation *f = kv.second;
      CC c = funcSummary(kv.first, 0);
      f->setAttr("kbcir.compose_worst", b.getI64IntegerAttr(c.worst));
      f->setAttr("kbcir.compose_expected", b.getI64IntegerAttr(c.expected));
      f->setAttr("kbcir.compose_reused", b.getI64IntegerAttr(c.reused));
      // dynamic-shape contract: true => the cost is a worst-case bound holding for any actual
      // size <= the declared count (compose.plan_holds_for), not an exact static plan.
      f->setAttr("kbcir.compose_dynamic", b.getBoolAttr(c.dynamic));
      if (hasBudget)
        f->setAttr("kbcir.compose_feasible", b.getBoolAttr(c.feasible));
      // alias/effect modeling: the function's read/write footprint + which sibling calls commute.
      Effect eff;
      regionEffect(f->getRegion(0).front(), emptySubst, eff, 0);
      f->setAttr("kbcir.effect_reads", symbolArray(eff.reads, b));
      f->setAttr("kbcir.effect_writes", symbolArray(eff.writes, b));
      annotateIndependence(f->getRegion(0).front(), emptySubst, b, 0);
    }
    if (invalidNesting) {
      root->emitError(
          "bcir-compose: recursive call graph or nesting deeper than 64 is unsupported");
      signalPassFailure();
    }
  }

  // A function's summary: its body planned ONCE over its formals (no substitution), memoized.
  CC funcSummary(StringRef name, int depth) {
    if (activeSummaries.count(name)) {
      invalidNesting = true;
      CC cyclic;
      cyclic.worst = std::numeric_limits<int64_t>::max();
      cyclic.expected = std::numeric_limits<int64_t>::max();
      cyclic.feasible = false;
      return cyclic;
    }
    auto it = memo.find(name);
    if (it != memo.end())
      return it->second;
    Operation *f = funcByName.lookup(name);
    if (!f || depth > 64) {
      if (depth > 64)
        invalidNesting = true;
      return {};
    }
    activeSummaries.insert(name);
    CC c = regionCost(f->getRegion(0).front(), depth, emptySubst);
    activeSummaries.erase(name);
    memo[name] = c;
    return c;
  }

  CC regionCost(Block &block, int depth, const llvm::DenseMap<StringRef, StringRef> &subst) {
    if (depth > 64) {
      invalidNesting = true;
      CC limited;
      limited.worst = std::numeric_limits<int64_t>::max();
      limited.expected = std::numeric_limits<int64_t>::max();
      limited.feasible = false;
      return limited;
    }
    CC acc;
    // Leaf: the region's direct bcir.claim ops, priced (with the substitution) as one block.
    SmallVector<ClaimOp> leaf;
    for (Operation &op : block)
      if (auto c = dyn_cast<ClaimOp>(&op))
        leaf.push_back(c);
    if (!leaf.empty()) {
      std::vector<cm::Column> cols = cm::fusedColumnsFromClaims(leaf, h, resByName, subst);
      int64_t total = 0;
      if (hasBudget) {
        bool feasible = true;
        total = cm::planConstrained(cols, w, theta, budgetDims, budgetCaps, feasible);
        if (!feasible)
          acc.feasible = false; // no realization of this block fits the caps (rcsp.Infeasible)
      } else {
        cm::planChosen(cols, w, theta, total);
      }
      acc.worst = saturatingAddNonnegative(acc.worst, total);
      acc.expected = saturatingAddNonnegative(acc.expected, total);
      acc.leaves = saturatingAddNonnegative(acc.leaves, 1);
      for (ClaimOp c : leaf)
        if (c.getDynamic())
          acc.dynamic = true; // a dynamic-shape claim -> count is a static upper bound
    }
    // Series composition with the nested control flow / calls.
    for (Operation &op : block) {
      if (auto cond = dyn_cast<KBCIRCondOp>(&op)) {
        CC t = regionCost(cond.getThenRegion().front(), depth + 1, subst);
        CC e = regionCost(cond.getElseRegion().front(), depth + 1, subst);
        int64_t p = std::max<int64_t>(
            0, std::min<int64_t>(1000, static_cast<int64_t>(cond.getProbThenMilli())));
        int64_t branchWorst = saturatingAddNonnegative(kPredCost, std::max(t.worst, e.worst));
        int64_t weighted = saturatingAddNonnegative(saturatingMulNonnegative(p, t.expected),
                                                    saturatingMulNonnegative(1000 - p, e.expected));
        int64_t branchExpected = saturatingAddNonnegative(kPredCost, weighted / 1000);
        acc.worst = saturatingAddNonnegative(acc.worst, branchWorst);
        acc.expected = saturatingAddNonnegative(acc.expected, branchExpected);
        acc.leaves =
            saturatingAddNonnegative(acc.leaves, saturatingAddNonnegative(t.leaves, e.leaves));
        acc.reused =
            saturatingAddNonnegative(acc.reused, saturatingAddNonnegative(t.reused, e.reused));
        acc.feasible = acc.feasible && t.feasible && e.feasible;
        acc.dynamic = acc.dynamic || t.dynamic || e.dynamic;
      } else if (auto call = dyn_cast<KBCIRCallOp>(&op)) {
        llvm::DenseMap<StringRef, StringRef> amap; // formal -> effective actual (composed)
        buildAmap(call, subst, amap);
        StringRef callee = call.getCallee();
        Operation *cf = funcByName.lookup(callee);
        if (!cf)
          continue; // undefined callee: R18 rejects it; contribute nothing
        if (summaryApplies(cf, amap)) {
          CC s = funcSummary(callee, depth + 1); // reuse the once-planned cost
          acc.worst = saturatingAddNonnegative(acc.worst, s.worst);
          acc.expected = saturatingAddNonnegative(acc.expected, s.expected);
          acc.reused =
              saturatingAddNonnegative(acc.reused, saturatingAddNonnegative(s.reused, s.leaves));
          acc.feasible = acc.feasible && s.feasible;
          acc.dynamic = acc.dynamic || s.dynamic;
        } else {
          CC inl = regionCost(cf->getRegion(0).front(), depth + 1, amap); // re-price inlined
          acc.worst = saturatingAddNonnegative(acc.worst, inl.worst);
          acc.expected = saturatingAddNonnegative(acc.expected, inl.expected);
          acc.leaves = saturatingAddNonnegative(acc.leaves, inl.leaves);
          acc.reused = saturatingAddNonnegative(acc.reused, inl.reused);
          acc.feasible = acc.feasible && inl.feasible;
          acc.dynamic = acc.dynamic || inl.dynamic;
        }
      }
    }
    return acc;
  }

  // Compose a call's (formal -> actual) map with the enclosing substitution (so an inner
  // call's actuals are themselves remapped -- compose._inline).
  void buildAmap(KBCIRCallOp call, const llvm::DenseMap<StringRef, StringRef> &subst,
                 llvm::DenseMap<StringRef, StringRef> &out) {
    ArrayAttr formals = call.getFormalsAttr();
    ArrayAttr actuals = call.getActualsAttr();
    if (!formals || !actuals)
      return;
    for (size_t i = 0; i < formals.size() && i < actuals.size(); ++i) {
      auto f = dyn_cast<FlatSymbolRefAttr>(formals[i]);
      auto a = dyn_cast<FlatSymbolRefAttr>(actuals[i]);
      if (!f || !a)
        continue;
      StringRef actualSym = a.getValue();
      auto it = subst.find(actualSym);
      if (it != subst.end())
        actualSym = it->second;
      out[f.getValue()] = actualSym;
    }
  }

  // compose._summary_applies: the summary's cost is exact for this call iff every formal the
  // callee touches has an actual with the same cost-key (a formal absent from the map maps to
  // itself, trivially matching).
  bool summaryApplies(Operation *calleeFunc, const llvm::DenseMap<StringRef, StringRef> &amap) {
    llvm::DenseSet<StringRef> footprint;
    calleeFunc->walk([&](ClaimOp c) {
      for (ArrayAttr refs : {c.getReads(), c.getWrites()})
        for (Attribute a : refs)
          if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
            footprint.insert(ref.getValue());
    });
    for (auto &kv : amap) {
      if (!footprint.count(kv.first))
        continue;
      ResourceOp rf = resByName.lookup(kv.first);
      ResourceOp ra = resByName.lookup(kv.second);
      if (!rf || !ra || !costKeyEq(rf, ra))
        return false;
    }
    return true;
  }

  // compose.effect: fold one op's read/write footprint into `out`, remapping symbols through
  // `subst` and recursing a kbcir.call into its callee (with the composed substitution).
  void opEffect(Operation *op, const llvm::DenseMap<StringRef, StringRef> &subst, Effect &out,
                int depth) {
    if (depth > 64)
      return;
    auto remap = [&](StringRef s) -> StringRef {
      auto it = subst.find(s);
      return it != subst.end() ? it->second : s;
    };
    if (auto c = dyn_cast<ClaimOp>(op)) {
      for (Attribute a : c.getReads())
        if (auto r = dyn_cast<FlatSymbolRefAttr>(a))
          out.reads.insert(remap(r.getValue()));
      for (Attribute a : c.getWrites())
        if (auto r = dyn_cast<FlatSymbolRefAttr>(a))
          out.writes.insert(remap(r.getValue()));
    } else if (auto cond = dyn_cast<KBCIRCondOp>(op)) {
      regionEffect(cond.getThenRegion().front(), subst, out, depth + 1);
      regionEffect(cond.getElseRegion().front(), subst, out, depth + 1);
    } else if (auto call = dyn_cast<KBCIRCallOp>(op)) {
      Operation *cf = funcByName.lookup(call.getCallee());
      if (!cf)
        return; // undefined callee: no statically-known footprint (oracle _EMPTY_EFFECT)
      llvm::DenseMap<StringRef, StringRef> amap;
      buildAmap(call, subst, amap);
      regionEffect(cf->getRegion(0).front(), amap, out, depth + 1);
    }
  }

  void regionEffect(Block &block, const llvm::DenseMap<StringRef, StringRef> &subst, Effect &out,
                    int depth) {
    for (Operation &op : block)
      opEffect(&op, subst, out, depth);
  }

  // Annotate each kbcir.call that has a preceding sibling with kbcir.commutes_with_prev =
  // independent(prev, this) -- their footprints are disjoint, so the two may be reordered or
  // overlapped (the cross-call alias test the pairwise plan cannot see).
  void annotateIndependence(Block &block, const llvm::DenseMap<StringRef, StringRef> &subst,
                            Builder &b, int depth) {
    if (depth > 64)
      return;
    Effect prevEff;
    bool havePrev = false;
    for (Operation &op : block) {
      Effect e;
      opEffect(&op, subst, e, depth);
      if (auto call = dyn_cast<KBCIRCallOp>(&op)) {
        if (havePrev)
          call->setAttr("kbcir.commutes_with_prev", b.getBoolAttr(!effectsConflict(prevEff, e)));
      } else if (auto cond = dyn_cast<KBCIRCondOp>(&op)) {
        annotateIndependence(cond.getThenRegion().front(), subst, b, depth + 1);
        annotateIndependence(cond.getElseRegion().front(), subst, b, depth + 1);
      }
      prevEff = std::move(e);
      havePrev = true;
    }
  }

  // A sorted (deterministic) FlatSymbolRef array for an effect footprint.
  ArrayAttr symbolArray(const llvm::DenseSet<StringRef> &syms, Builder &b) {
    SmallVector<StringRef> v(syms.begin(), syms.end());
    std::sort(v.begin(), v.end());
    SmallVector<Attribute> refs;
    for (StringRef s : v)
      refs.push_back(FlatSymbolRefAttr::get(&getContext(), s));
    return b.getArrayAttr(refs);
  }
};

} // namespace

std::unique_ptr<Pass> createComposePass() {
  return std::make_unique<ComposePass>();
}

} // namespace bcir
