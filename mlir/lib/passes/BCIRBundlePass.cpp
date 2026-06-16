//===- BCIRBundlePass.cpp - detect multi-claim (joint) bundles ----*- C++ -*-===//
//
// -bcir-bundle: the law-rail port of kbcir.bundle's *analysis*. It finds the clusters of
// mutually-independent same-phase claims that share a read operand -- the bundles whose
// joint intra-phase reorder can recover a fusion discount the pairwise shortest path
// misses (see kbcir/bundle.py). This pass surfaces that structure on the IR: it annotates
// each bundled claim with `kbcir.bundle` (the bundle index) + `kbcir.bundle_shared` (the
// shared resource) and the enclosing bcir.module with `kbcir.bundle_count`.
//
// Read-only: it does NOT reorder claims or re-price the plan. The joint-gain computation
// (reordering the cost-model columns and re-running the shortest path) and the reorder
// transformation are the tracked next increment; detection is the first port.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRCostModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <string>

using namespace mlir;

namespace bcir {
namespace {

// RAW/WAR/WAW between two claims' read/write symbol sets (mirrors bundle._conflict).
static bool conflict(ArrayRef<StringRef> ar, ArrayRef<StringRef> aw,
                     ArrayRef<StringRef> br, ArrayRef<StringRef> bw) {
  auto intersects = [](ArrayRef<StringRef> x, ArrayRef<StringRef> y) {
    for (StringRef a : x)
      for (StringRef b : y)
        if (a == b)
          return true;
    return false;
  };
  return intersects(aw, br) || intersects(aw, bw) || intersects(bw, ar);
}

struct BundlePass : public PassWrapper<BundlePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BundlePass)

  StringRef getArgument() const final { return "bcir-bundle"; }
  StringRef getDescription() const final {
    return "Detect multi-claim (joint) bundles -- clusters of mutually-independent "
           "same-phase claims sharing a read operand (the law-rail analysis of "
           "kbcir.bundle); annotates kbcir.bundle / bundle_shared / bundle_count.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b);
    });
  }

  void runOnModule(Operation *root, Builder &b) {
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

    // Unique phase ids in first-appearance order (avoid a pair-keyed DenseMap).
    SmallVector<int32_t> phases;
    for (const Info &in : claims)
      if (std::find(phases.begin(), phases.end(), in.phase) == phases.end())
        phases.push_back(in.phase);

    // Per phase, per shared read symbol: the maximal mutually-independent set of claims
    // that read it (size >= 2). Deduplicated by claim set; mirrors bundle.find_bundles.
    llvm::StringSet<> seenSets;
    int bundleIdx = 0;
    for (int32_t ph : phases) {
      llvm::MapVector<StringRef, SmallVector<unsigned>> byRead;  // StringRef key (safe)
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
          bool ok = true;
          for (unsigned k : indep)
            if (conflict(claims[m].reads, claims[m].writes, claims[k].reads,
                         claims[k].writes)) {
              ok = false;
              break;
            }
          if (ok)
            indep.push_back(m);
        }
        if (indep.size() < 2)
          continue;
        std::string sig;  // dedup by the claim-id set
        for (unsigned m : indep) {
          sig += std::to_string(claims[m].claim.getClaimId());
          sig += ",";
        }
        if (!seenSets.insert(sig).second)
          continue;
        for (unsigned m : indep) {
          claims[m].claim->setAttr("kbcir.bundle", b.getI64IntegerAttr(bundleIdx));
          claims[m].claim->setAttr(
              "kbcir.bundle_shared", FlatSymbolRefAttr::get(&getContext(), kv.first));
        }
        ++bundleIdx;
      }
    }

    root->setAttr("kbcir.bundle_count", b.getI64IntegerAttr(bundleIdx));
  }
};

} // namespace

std::unique_ptr<Pass> createBundlePass() { return std::make_unique<BundlePass>(); }

} // namespace bcir
