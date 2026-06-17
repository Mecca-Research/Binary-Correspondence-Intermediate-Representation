//===- BCIRAllocPoolPass.cpp - liveness-based memory pool planning -*- C++ -*-===//
//
// -bcir-alloc-pool: the law-rail port of kbcir.allocator.pool_plan -- predictive pool
// allocation. Resources whose live ranges (the [first_phase, last_phase] span over the phase
// order) do NOT overlap can share one memory arena, so the peak footprint (sum of arena sizes)
// drops below the naive sum of every tensor -- defragmenting *before* the kernel runs.
//
// Greedy left-edge interval partitioning: each touched resource, in (start, rid) order, reuses
// the first arena whose last member has already died (its last_phase < this resource's
// first_phase), else opens a new one. Deterministic and gains-only (peak <= naive always; a
// single-phase program where everything is simultaneously live is a clean no-op).
//
// Annotates each pooled resource with kbcir.pool_id and the module with kbcir.pool_naive_bytes
// / pool_peak_bytes / pool_saved. Reproduces pool_plan bit-for-bit. A modeled footprint gain.
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

struct AllocPoolPass : public PassWrapper<AllocPoolPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AllocPoolPass)

  StringRef getArgument() const final { return "bcir-alloc-pool"; }
  StringRef getDescription() const final {
    return "Liveness-based memory pool planning (kbcir.allocator.pool_plan): resources with "
           "disjoint live ranges share an arena (greedy left-edge); annotates kbcir.pool_id "
           "per resource + kbcir.pool_naive_bytes / pool_peak_bytes / pool_saved.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b);
    });
  }

  void runOnModule(Operation *root, Builder &b) {
    // Element width for sizing (the capability's, or the model default 4 B).
    int64_t elemBytes = 4;
    if (auto cap = cm::firstCapability(root))
      elemBytes = cm::readCap(cap).elemBytes;
    auto resByName = cm::resourcesByName(root);

    // Phase order positions (declaration order; allocator.live_intervals keys on this).
    llvm::DenseMap<StringRef, int64_t> phasePos;
    int64_t pos = 0;
    root->walk([&](PhaseOp p) { phasePos[p.getSymName()] = pos++; });

    // Live interval [lo, hi] per touched resource symbol (the lineage span).
    llvm::DenseMap<StringRef, std::pair<int64_t, int64_t>> span;
    root->walk([&](ClaimOp c) {
      int64_t cpos = phasePos.lookup(c.getPhase());
      for (ArrayAttr refs : {c.getReads(), c.getWrites()})
        for (Attribute a : refs)
          if (auto ref = dyn_cast<FlatSymbolRefAttr>(a)) {
            StringRef sym = ref.getValue();
            auto it = span.find(sym);
            if (it == span.end())
              span[sym] = {cpos, cpos};
            else {
              it->second.first = std::min(it->second.first, cpos);
              it->second.second = std::max(it->second.second, cpos);
            }
          }
    });

    // Resolve to (rid, [lo,hi], size) and sort by (start, rid) -- the left-edge order.
    struct Res {
      StringRef sym;
      int64_t rid, lo, hi, size;
    };
    SmallVector<Res> rs;
    for (auto &kv : span) {
      ResourceOp r = resByName.lookup(kv.first);
      if (!r)
        continue;
      int64_t cnt = 1;
      for (int64_t d : r.getShape())
        cnt *= (d > 0 ? d : 1);
      rs.push_back(
          {kv.first, static_cast<int64_t>(r.getRid()), kv.second.first, kv.second.second,
           cnt * elemBytes});
    }
    std::sort(rs.begin(), rs.end(), [](const Res &a, const Res &z) {
      return a.lo != z.lo ? a.lo < z.lo : a.rid < z.rid;
    });

    // Greedy left-edge partitioning: reuse the first arena whose last member has died.
    struct Arena {
      int64_t last, size;
    };
    SmallVector<Arena> arenas;
    llvm::DenseMap<StringRef, int64_t> poolOf;
    int64_t naive = 0;
    for (const Res &r : rs) {
      naive += r.size;
      int slot = -1;
      for (int i = 0; i < static_cast<int>(arenas.size()); ++i)
        if (arenas[i].last < r.lo) {
          slot = i;
          break;
        }
      if (slot < 0) {
        arenas.push_back({r.hi, r.size});
        poolOf[r.sym] = static_cast<int64_t>(arenas.size()) - 1;
      } else {
        arenas[slot].last = r.hi;
        arenas[slot].size = std::max(arenas[slot].size, r.size);
        poolOf[r.sym] = slot;
      }
    }
    int64_t peak = 0;
    for (const Arena &a : arenas)
      peak += a.size;

    root->walk([&](ResourceOp r) {
      auto it = poolOf.find(r.getSymName());
      if (it != poolOf.end())
        r->setAttr("kbcir.pool_id", b.getI64IntegerAttr(it->second));
    });
    root->setAttr("kbcir.pool_naive_bytes", b.getI64IntegerAttr(naive));
    root->setAttr("kbcir.pool_peak_bytes", b.getI64IntegerAttr(peak));
    root->setAttr("kbcir.pool_saved", b.getI64IntegerAttr(naive - peak));
  }
};

} // namespace

std::unique_ptr<Pass> createAllocPoolPass() { return std::make_unique<AllocPoolPass>(); }

} // namespace bcir
