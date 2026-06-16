//===- BCIRCostModel.cpp - the K_BCIR cost algebra on the MLIR rail -*- C++ -*-===//
//
// -bcir-cost-model: the keystone of the optimizer-core port. It recomputes each
// claim's candidate set and 12-d cost vectors FROM FIRST PRINCIPLES -- the claim
// graph + the target capability -- instead of trusting the emitter-baked
// bcir.kbcir.path costs. A faithful C++23 port of bcir/kbcir/cost.py::_cost +
// realize.candidates_for + _stride_penalty, with the seeded constants read from
// bcir.target.capability and the memory-tier Q8 factors as a constexpr table
// (mirroring cost.py MemoryHierarchy.default()).
//
// Annotates each claim with kbcir.cm_candidates (count), kbcir.cm_min_cost (the
// argmin candidate's base cost vector), and kbcir.cm_min_score (its scalarized
// score under the first declared policy). On the canonical vector_add it recomputes
// the vec16 cost <compute=64, memory=3840, thermal=1088, power=1088> and the score
// 7808 from the claim alone -- the law no longer depends on the Python emitter.
//
// Scope note: candidate *enumeration* keys on the BCIR op-string vocabulary
// (linalg.matmul -> tile op-class 2; histogram.scatter -> gather op-class 0;
// reduce.gather -> the blocked/gather pair; else the elementwise op-class 1), the
// MLIR analog of cost.py's opcode dispatch. The cross-claim coupling
// (fusion/CSE/deforestation + the layered min-plus path) is the next ports
// (BCIR_LOWERING_PLAN.md steps 2-3); this pass establishes the base cost algebra.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <array>
#include <cstdint>
#include <string>

using namespace mlir;

namespace bcir {
namespace {

using Cost = std::array<int64_t, 12>; // DIMS order (see BCIRAttrs.td / cost.py)

// Memory-tier Q8 factors keyed by the resource domain -- a constexpr port of
// cost.py MemoryHierarchy.default() composed with the Domain->MemTier map. {bw, lat}
// relative to DRAM (256 == x1.0); lower bw = more bandwidth (cheaper).
struct TierQ8 { int64_t bw, lat; };
constexpr TierQ8 tierForDomain(Domain d) {
  switch (d) {
  case Domain::RAM:  return {256, 256};   // DRAM (x1.0 baseline)
  case Domain::MMIO: return {256, 256};   // DRAM-class
  case Domain::VRAM: return {64, 192};    // HBM
  case Domain::HBM:  return {64, 192};    // HBM
  case Domain::CXL:  return {384, 512};   // memory-semantic, costlier than DRAM
  case Domain::NVM:  return {1024, 4096}; // SSD-class swap backing
  }
  return {256, 256};
}

// The seeded target constants (cost.py TargetProfile), read off the capability op.
struct Cap {
  SmallVector<int64_t> widths;  // sorted unique lane widths
  int64_t gatherPenalty, cacheline, memUnit, baseOverhead;
  int64_t thermalDensity, powerDensity, perOpHeat, elemBytes;
};

// ceil(log2 n) -- HAM turns random access into O(log n) (cost.py: (n-1).bit_length()).
static int64_t bitLength(int64_t n) {
  int64_t bits = 0;
  for (uint64_t v = static_cast<uint64_t>(n); v; v >>= 1)
    ++bits;
  return bits;
}

// cost.py::_opclass via the BCIR op-string vocabulary.
static int opClassFor(StringRef op) {
  if (op == "linalg.matmul")
    return 2; // T_MACC
  if (op.starts_with("histogram") || op.contains("scatter"))
    return 0; // GGG_LOAD/STORE
  return 1;   // ADD / SUB / MUL (incl. vector.* and reduce.*)
}

// cost.py::_stride_penalty.
static int64_t stridePenalty(StrideClass sc, int strideK, const Cap &h) {
  switch (sc) {
  case StrideClass::Unit:
  case StrideClass::Scalar:
  case StrideClass::Tile:
    return 1;
  case StrideClass::Strided:
    return std::min<int64_t>(std::max<int64_t>(strideK, 1),
                             h.cacheline / std::max<int64_t>(1, h.elemBytes));
  case StrideClass::Cacheline:
    return 2;
  case StrideClass::Random:
    return h.gatherPenalty;
  }
  return 1;
}

// cost.py::_cost -- the 12-d base cost of realizing `count` elements at `width`.
static Cost baseCost(int64_t count, int streams, int opclass, int width,
                     int64_t penalty, TierQ8 tier, const Cap &h,
                     int64_t extraCompile = 0) {
  int64_t n = std::max<int64_t>(1, count);
  int64_t ceil = (n + width - 1) / width;
  Cost c{};
  c[0] = ceil * opclass;                                    // compute
  int64_t memShared = (n * streams * h.memUnit * tier.bw) >> 8;
  int64_t accessOps = ceil * streams;
  int64_t overhead = h.baseOverhead * penalty;
  c[1] = memShared + ((accessOps * overhead * tier.lat) >> 8); // memory
  c[4] = extraCompile;                                      // compile
  c[5] = static_cast<int64_t>(width) * h.thermalDensity + ceil * h.perOpHeat; // thermal
  c[6] = static_cast<int64_t>(width) * h.powerDensity + ceil * h.perOpHeat;   // power
  return c;
}

// realize.candidates_for -- the legal realizations' base cost vectors (the candidate
// names are not annotated, so only the costs are returned, in declared order).
static SmallVector<Cost> candidatesFor(ClaimOp c, const Cap &h, Domain dom,
                                       bool ham) {
  StringRef op = c.getOp();
  StrideClass sc = c.getStrideClass();
  int64_t count = static_cast<int64_t>(c.getCount());
  int streams = static_cast<int>(c.getReads().size() + c.getWrites().size());
  int opclass = opClassFor(op);
  TierQ8 tier = tierForDomain(dom);
  int64_t gp = ham ? std::max<int64_t>(1, bitLength(count - 1)) : h.gatherPenalty;

  SmallVector<Cost> out;
  if (op == "reduce.gather") {                       // blocked vs gather
    out.push_back(baseCost(count, streams, opclass, 1, 1, tier, h));
    out.push_back(baseCost(count, streams, opclass, 1, gp, tier, h));
    return out;
  }
  switch (sc) {
  case StrideClass::Unit:
  case StrideClass::Scalar:
    for (int64_t w : h.widths)                       // scalar + vec{w}
      out.push_back(baseCost(count, streams, opclass, static_cast<int>(w), 1, tier, h));
    break;
  case StrideClass::Strided:
    out.push_back(baseCost(count, streams, opclass, 1,
                           stridePenalty(sc, c.getStrideK(), h), tier, h));
    out.push_back(baseCost(count, streams, opclass, 1, gp, tier, h));
    break;
  case StrideClass::Cacheline: {
    int64_t uxw = std::min<int64_t>(8, h.widths.back());
    out.push_back(baseCost(count, streams, opclass, static_cast<int>(uxw), 2, tier, h,
                           count / 4));
    out.push_back(baseCost(count, streams, opclass, 1, gp, tier, h));
    break;
  }
  case StrideClass::Random:
    out.push_back(baseCost(count, streams, opclass, 1, gp, tier, h));
    break;
  case StrideClass::Tile:
    out.push_back(baseCost(count, streams, opclass, 16, 1, tier, h));
    break;
  }
  return out;
}

static int64_t scalarizeCost(const Cost &c, ArrayRef<int64_t> w) {
  int64_t s = 0;
  for (int i = 0; i < 12 && i < static_cast<int>(w.size()); ++i)
    s += c[i] * w[i];
  return s;
}

// Intra-phase redundancy factors (realize.fused_candidates / _context_factor's
// dependency-based half), applied to every candidate of a claim in Q8 (256 == x1.0).
using Factor = std::array<int64_t, 12>;
constexpr Factor kIdentity = {256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256};

// producer->consumer deforestation: the intermediate never round-trips -> x0.75 memory.
static Factor deforestFactor() {
  Factor f = kIdentity;
  f[1] = 192;
  return f;
}

// CSE / duplicate elimination: the value is already computed, so no recompute
// (compute zeroed) and only a copy of the result is written instead of re-reading
// every operand (memory scaled from `nrd+nwr` streams to the `1+nwr` a copy needs).
static Factor cseFactor(int nrd, int nwr) {
  int64_t full = static_cast<int64_t>(nrd) + nwr, copy = 1 + nwr;
  Factor f = kIdentity;
  f[0] = 0;
  f[1] = (copy * 256) / std::max<int64_t>(1, full);
  return f;
}

static void applyFactor(Cost &c, const Factor &f) {
  for (int i = 0; i < 12; ++i)
    c[i] = (c[i] * f[i]) >> 8;
}

struct CostModelPass : public PassWrapper<CostModelPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CostModelPass)

  StringRef getArgument() const final { return "bcir-cost-model"; }
  StringRef getDescription() const final {
    return "Recompute each claim's candidate costs from the claim + target "
           "capability (the K_BCIR cost algebra, C++23 port of cost.py); annotates "
           "kbcir.cm_candidates / cm_min_cost / cm_min_score.";
  }

  void runOnOperation() override {
    Operation *root = getOperation();
    Builder b(&getContext());

    // The first target.capability supplies the seeded constants.
    TargetCapabilityOp capOp;
    root->walk([&](TargetCapabilityOp t) {
      if (!capOp)
        capOp = t;
    });
    if (!capOp)
      return; // nothing to compute against (no target declared)

    Cap h;
    {
      llvm::SmallSetVector<int64_t, 4> ws;
      for (int64_t w : capOp.getLaneWidths())
        ws.insert(w);
      h.widths.assign(ws.begin(), ws.end());
      llvm::sort(h.widths);
      if (h.widths.empty())
        h.widths.push_back(1);
      h.gatherPenalty = capOp.getGatherPenalty();
      h.cacheline = capOp.getCacheline();
      h.memUnit = capOp.getMemUnit();
      h.baseOverhead = capOp.getBaseOverhead();
      h.thermalDensity = capOp.getThermalDensity();
      h.powerDensity = capOp.getPowerDensity();
      h.perOpHeat = capOp.getPerOpHeat();
      h.elemBytes = capOp.getElemBytes();
    }

    llvm::DenseMap<StringRef, ResourceOp> resByName;
    root->walk([&](ResourceOp r) { resByName[r.getSymName()] = r; });

    // The first policy's weights scalarize the candidate costs (theta-folded by the
    // emitter; absent -> annotate costs without a score).
    ArrayRef<int64_t> weights;
    root->walk([&](KBCIRPolicyOp p) {
      if (weights.empty())
        weights = p.getWeights();
    });

    // Phase id for ordering (topological by id -- the SchedulePass convention);
    // claims are processed in (phase id, declared) order, the fused_candidates walk.
    llvm::DenseMap<StringRef, int32_t> phaseId;
    root->walk([&](PhaseOp p) { phaseId[p.getSymName()] = p.getId(); });
    SmallVector<ClaimOp> claims;
    root->walk([&](ClaimOp c) { claims.push_back(c); });
    llvm::stable_sort(claims, [&](ClaimOp a, ClaimOp z) {
      return phaseId.lookup(a.getPhase()) < phaseId.lookup(z.getPhase());
    });

    // Per-phase intra-phase data-flow state (reset at a phase boundary -- a barrier
    // materializes intermediates, so the credits are intra-phase only).
    int32_t curPhase = 0;
    bool havePhase = false;
    llvm::DenseSet<StringRef> produced;        // rids written so far (deforestation)
    llvm::DenseMap<StringRef, int> version;    // value numbering (a write bumps it)
    llvm::StringSet<> seen;                     // value-numbered compute signatures

    auto refs = [](ArrayAttr a, SmallVectorImpl<StringRef> &out) {
      for (Attribute x : a)
        if (auto r = dyn_cast<FlatSymbolRefAttr>(x))
          out.push_back(r.getValue());
    };

    for (ClaimOp c : claims) {
      int32_t pid = phaseId.lookup(c.getPhase());
      if (!havePhase || pid != curPhase) {
        produced.clear();
        version.clear();
        seen.clear();
        curPhase = pid;
        havePhase = true;
      }

      SmallVector<StringRef> reads, writes;
      refs(c.getReads(), reads);
      refs(c.getWrites(), writes);

      // Cost resource = the first read (cost.py: rd[0]); its domain selects the tier
      // and its access flags HAM.
      Domain dom = c.getDomain();
      bool ham = false;
      if (!reads.empty())
        if (auto r = resByName.lookup(reads[0])) {
          dom = r.getDomainKind();
          ham = r.getAccess() && *r.getAccess() == Access::HAM;
        }

      SmallVector<Cost> cands = candidatesFor(c, h, dom, ham);
      if (cands.empty())
        continue;

      // value-numbered compute signature: op + (read, version) pairs.
      std::string sig = c.getOp().str();
      for (StringRef rd : reads) {
        sig += "|";
        sig += rd.str();
        sig += "@";
        sig += std::to_string(version.lookup(rd));
      }
      bool cse = !reads.empty() && seen.count(sig);
      bool defo = false;
      if (!cse)
        for (StringRef rd : reads)
          if (produced.count(rd)) {
            defo = true;
            break;
          }
      if (cse || defo) {
        Factor f = cse ? cseFactor(static_cast<int>(reads.size()),
                                   static_cast<int>(writes.size()))
                       : deforestFactor();
        for (Cost &cc : cands)
          applyFactor(cc, f);
      }

      // Bookkeeping: first occurrence records the signature; writes bump versions.
      if (!reads.empty())
        seen.insert(sig);
      for (StringRef w : writes)
        version[w] = version.lookup(w) + 1;
      for (StringRef w : writes)
        produced.insert(w);

      // Annotate the computed candidate count, the argmin cost + score, and which
      // redundancy credit (if any) the claim earned.
      c->setAttr("kbcir.cm_candidates",
                 b.getI64IntegerAttr(static_cast<int64_t>(cands.size())));
      if (cse)
        c->setAttr("kbcir.cm_fusion", b.getStringAttr("cse"));
      else if (defo)
        c->setAttr("kbcir.cm_fusion", b.getStringAttr("deforest"));

      size_t best = 0;
      if (!weights.empty()) {
        int64_t bestScore = scalarizeCost(cands[0], weights);
        for (size_t i = 1; i < cands.size(); ++i) {
          int64_t s = scalarizeCost(cands[i], weights);
          if (s < bestScore) {
            bestScore = s;
            best = i;
          }
        }
        c->setAttr("kbcir.cm_min_score", b.getI64IntegerAttr(bestScore));
      }
      const Cost &bc = cands[best];
      c->setAttr("kbcir.cm_min_cost",
                 CostVectorAttr::get(&getContext(), bc[0], bc[1], bc[2], bc[3], bc[4],
                                     bc[5], bc[6], bc[7], bc[8], bc[9], bc[10], bc[11]));
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createCostModelPass() {
  return std::make_unique<CostModelPass>();
}

}  // namespace bcir
