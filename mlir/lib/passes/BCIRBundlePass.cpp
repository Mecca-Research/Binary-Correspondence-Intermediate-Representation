//===- BCIRBundlePass.cpp - detect + joint-reorder multi-claim bundles -*- C++ -*-===//
//
// -bcir-bundle: the law-rail port of kbcir.bundle (bcir/kbcir/bundle.py). It finds the
// clusters of mutually-independent same-phase claims that share a read operand -- the
// bundles whose joint intra-phase reorder can recover a fusion discount the pairwise
// shortest path misses -- and then prices that joint reorder.
//
//   detection  : annotate each bundled claim with `kbcir.bundle` (the bundle index) +
//                `kbcir.bundle_shared` (the shared resource); the module with
//                `kbcir.bundle_count`. Pure read/write analysis (no cost model needed).
//
//   joint-reorder (the transformation): when a target.capability + policy are present,
//                reorder the *cost-model columns* (BCIRCostModel.h fusedColumns) so the
//                bundle's members are contiguous, re-run the min-plus shortest path
//                (cm::planChosen) for every legal intra-bundle ordering, and annotate the
//                gain over the declared-order plan as `kbcir.bundle_gain` + the recommended
//                `kbcir.bundle_order` (claim-id order) + `kbcir.bundle_searched`. This is
//                the column-reorder + re-price of bundle.optimize_bundled -- the members are
//                mutually independent, so every permutation is a legal linear extension and
//                only the path-based shared-input coupling changes; it never mutates the IR.
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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace mlir;

namespace bcir {
namespace {

// ASM3b + §5.14 Phase 2 (mirrors bundle._conflict's fence clause): a `barriered` claim is an
// ordering fence and a VOLATILE access (MMIO) must never be reordered/bundled across anything --
// both conflict with every other claim, so no bundle ever forms across them. This closes a live
// oracle<->law divergence: the Python bundle._conflict has fenced `barriered` since ASM3b, but
// this pass's RAW/WAR/WAW check did not.
static bool fenced(ClaimOp c) {
  return c.getHazard() == HazardMode::Barriered || c.getIsVolatile();
}

// RAW/WAR/WAW between two claims' read/write symbol sets (mirrors bundle._conflict).
static bool conflict(ArrayRef<StringRef> ar, ArrayRef<StringRef> aw, ArrayRef<StringRef> br,
                     ArrayRef<StringRef> bw) {
  auto intersects = [](ArrayRef<StringRef> x, ArrayRef<StringRef> y) {
    for (StringRef a : x)
      for (StringRef b : y)
        if (a == b)
          return true;
    return false;
  };
  return intersects(aw, br) || intersects(aw, bw) || intersects(bw, ar);
}

// A copy of `cols` with the columns at `memberCols` removed from their positions and
// re-inserted contiguously, in `orderCols`, at the anchor (the earliest member position).
// Non-member columns keep their relative order. Mirrors bundle._reordered at the
// cost-column level -- the joint reorder we re-price (no IR is touched).
static std::vector<cm::Column> reorderColumns(const std::vector<cm::Column> &cols,
                                              ArrayRef<unsigned> memberCols,
                                              ArrayRef<unsigned> orderCols) {
  llvm::DenseSet<unsigned> member(memberCols.begin(), memberCols.end());
  unsigned anchor = *std::min_element(memberCols.begin(), memberCols.end());
  std::vector<cm::Column> out;
  out.reserve(cols.size());
  for (unsigned i = 0; i < cols.size(); ++i) {
    if (i == anchor)
      for (unsigned c : orderCols)
        out.push_back(cols[c]);
    if (!member.count(i))
      out.push_back(cols[i]);
  }
  return out;
}

struct BundlePass : public PassWrapper<BundlePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BundlePass)

  StringRef getArgument() const final { return "bcir-bundle"; }
  StringRef getDescription() const final {
    return "Detect + joint-reorder multi-claim (joint) bundles -- clusters of mutually-"
           "independent same-phase claims sharing a read operand (the law-rail port of "
           "kbcir.bundle); annotates kbcir.bundle / bundle_shared / bundle_count and, with a "
           "capability + policy, the re-priced kbcir.bundle_gain / bundle_order.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b, getChildAnalysis<cm::PlanAnalysis>(mod));
    });
    markAnalysesPreserved<cm::PlanAnalysis>();
  }

  void runOnModule(Operation *root, Builder &b, const cm::PlanAnalysis &pa) {
    // phase symbol -> phase id (so claims are grouped by the actual phase).
    llvm::DenseMap<StringRef, int32_t> phaseId;
    root->walk([&](PhaseOp p) { phaseId[p.getSymName()] = p.getId(); });

    // Collect each claim's reads/writes once.
    struct Info {
      ClaimOp claim;
      int32_t phase;
      SmallVector<StringRef> reads, writes;
    };
    SmallVector<Info> claims;
    root->walk([&](ClaimOp c) {
      Info in;
      in.claim = c;
      in.phase = phaseId.lookup(c.getPhase());
      cm::symRefs(c.getReads(), in.reads);
      cm::symRefs(c.getWrites(), in.writes);
      claims.push_back(std::move(in));
    });

    // The cost model (for the joint-reorder re-price), if a capability + policy are present.
    // Detection is pure read/write analysis and runs even without these.
    // A mutable copy of the shared plan's columns (Bundle re-prices permuted copies of them);
    // sourcing cols + the base total from PlanAnalysis avoids re-walking + re-planning here.
    std::vector<cm::Column> cols;
    llvm::DenseMap<Operation *, unsigned> colOf; // claim op -> column index
    ArrayRef<int64_t> w;
    int64_t theta = 0, baseTotal = 0;
    bool haveCost = false;
    if (pa.valid) {
      cols = pa.cols;
      w = pa.weights;
      theta = pa.thetaThermal;
      baseTotal = pa.total;
      for (unsigned i = 0; i < cols.size(); ++i)
        colOf[cols[i].claim.getOperation()] = i;
      haveCost = true;
    }

    // Unique phase ids in first-appearance order (avoid a pair-keyed DenseMap).
    SmallVector<int32_t> phases;
    for (const Info &in : claims)
      if (std::find(phases.begin(), phases.end(), in.phase) == phases.end())
        phases.push_back(in.phase);

    // Per phase, per shared read symbol: the maximal mutually-independent set of claims
    // that read it (size >= 2). Deduplicated by claim set; mirrors bundle.find_bundles.
    llvm::StringSet<> seenSets;
    int bundleIdx = 0;
    int64_t totalGain = 0;
    for (int32_t ph : phases) {
      llvm::MapVector<StringRef, SmallVector<unsigned>> byRead; // StringRef key (safe)
      for (unsigned i = 0; i < claims.size(); ++i)
        if (claims[i].phase == ph)
          for (StringRef rd : claims[i].reads)
            byRead[rd].push_back(i);

      for (auto &kv : byRead) {
        if (kv.second.size() < 2)
          continue;
        // Maximal mutually-independent subset, in claim_id order (stable).
        SmallVector<unsigned> members(kv.second.begin(), kv.second.end());
        llvm::stable_sort(members, [&](unsigned a, unsigned z) {
          return claims[a].claim.getClaimId() < claims[z].claim.getClaimId();
        });
        SmallVector<unsigned> indep;
        for (unsigned m : members) {
          bool ok = !fenced(claims[m].claim); // a fenced claim joins no bundle
          for (unsigned k : indep)
            if (!ok ||
                conflict(claims[m].reads, claims[m].writes, claims[k].reads, claims[k].writes)) {
              ok = false;
              break;
            }
          if (ok)
            indep.push_back(m);
        }
        if (indep.size() < 2)
          continue;
        std::string sig; // dedup by the claim-id set
        for (unsigned m : indep) {
          sig += std::to_string(claims[m].claim.getClaimId());
          sig += ",";
        }
        if (!seenSets.insert(sig).second)
          continue;

        // §5.14 Phase 2 / ASM3b parity with bundle._legal_reorder: making the members
        // contiguous moves them across every column inside the member span -- illegal across
        // a FENCE (a barriered/volatile claim never has another claim reordered across it).
        // Skip the re-price when a fenced claim sits inside the span: detection stands, only
        // the reorder gain is forfeited (the oracle's per-permutation legality check rejects
        // exactly those orders).
        bool fencedInSpan = false;
        if (haveCost) {
          unsigned lo = ~0u, hi = 0;
          for (unsigned m : indep) {
            unsigned c = colOf.lookup(claims[m].claim.getOperation());
            lo = std::min(lo, c);
            hi = std::max(hi, c);
          }
          for (unsigned c = lo + 1; c < hi; ++c)
            if (fenced(cols[c].claim))
              fencedInSpan = true;
        }

        // The joint-reorder re-price: reorder the cost columns so the members are
        // contiguous and re-run the shortest path for every intra-bundle ordering (the
        // members are mutually independent, so every permutation is legal). gain is the
        // win over the declared-order plan; >= 0 because the baseline is the floor.
        int64_t gain = 0;
        int searched = 0;
        SmallVector<int64_t> bestOrder;
        if (haveCost && indep.size() <= 6 && !fencedInSpan) {
          SmallVector<unsigned> memCol;
          for (unsigned m : indep)
            memCol.push_back(colOf.lookup(claims[m].claim.getOperation()));
          SmallVector<unsigned> perm(memCol.begin(), memCol.end());
          llvm::sort(perm); // deterministic next_permutation enumeration
          int64_t best = baseTotal;
          for (unsigned c : memCol)
            bestOrder.push_back(cols[c].claim.getClaimId()); // declared-order default
          do {
            std::vector<cm::Column> reordered = reorderColumns(cols, memCol, perm);
            int64_t t = 0;
            cm::planChosen(reordered, w, theta, t);
            ++searched;
            if (t < best) {
              best = t;
              bestOrder.clear();
              for (unsigned c : perm)
                bestOrder.push_back(cols[c].claim.getClaimId());
            }
          } while (std::next_permutation(perm.begin(), perm.end()));
          gain = baseTotal - best;
          totalGain = saturatingAddNonnegative(totalGain, gain);
        }

        for (unsigned m : indep) {
          ClaimOp claim = claims[m].claim;
          claim->setAttr("kbcir.bundle", b.getI64IntegerAttr(bundleIdx));
          claim->setAttr("kbcir.bundle_shared", FlatSymbolRefAttr::get(&getContext(), kv.first));
          if (haveCost && indep.size() <= 6 && !fencedInSpan) {
            claim->setAttr("kbcir.bundle_gain", b.getI64IntegerAttr(gain));
            claim->setAttr("kbcir.bundle_order", b.getI64ArrayAttr(bestOrder));
            claim->setAttr("kbcir.bundle_searched", b.getI64IntegerAttr(searched));
          }
        }
        ++bundleIdx;
      }
    }

    root->setAttr("kbcir.bundle_count", b.getI64IntegerAttr(bundleIdx));
    if (haveCost)
      root->setAttr("kbcir.bundle_total_gain", b.getI64IntegerAttr(totalGain));
  }
};

} // namespace

std::unique_ptr<Pass> createBundlePass() {
  return std::make_unique<BundlePass>();
}

} // namespace bcir
