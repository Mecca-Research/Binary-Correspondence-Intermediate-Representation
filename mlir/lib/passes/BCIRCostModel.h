//===- BCIRCostModel.h - shared K_BCIR cost algebra + fusion ------*- C++ -*-===//
//
// The C++23 port of bcir/kbcir/cost.py + realize.{candidates_for,fused_candidates},
// shared (header-only, inline) by the optimizer-core passes:
//   -bcir-cost-model  (BCIRCostModel.cpp)  : annotate per-claim candidate costs
//   -bcir-plan        (BCIRPlanPass.cpp)   : the layered min-plus shortest path
//
// `fusedColumns()` returns, in (phase, declared) order, each claim's candidate
// column with the intra-phase redundancy credits (deforestation / CSE) already
// baked in -- the path-INdependent half of the cost. The cross-claim
// `_context_factor` (path-based shared-input fusion) is applied per edge by the plan
// pass. NOT a public API.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_LIB_PASSES_BCIRCOSTMODEL_H
#define BCIR_LIB_PASSES_BCIRCOSTMODEL_H

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace mlir;

namespace bcir {
namespace cm {

using Cost = std::array<int64_t, 12>; // DIMS order (BCIRAttrs.td / cost.py)
using Factor = std::array<int64_t, 12>;
constexpr Factor kIdentity = {256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256};

// A realization candidate: its lane width (for the path-based context factor) + cost.
struct Cand {
  int64_t width;
  Cost cost;
};

// Memory-tier Q8 factors keyed by the resource domain (constexpr port of cost.py
// MemoryHierarchy.default() composed with Domain->MemTier). {bw, lat} vs DRAM (x1.0).
struct TierQ8 { int64_t bw, lat; };
constexpr TierQ8 tierForDomain(Domain d) {
  switch (d) {
  case Domain::RAM:  return {256, 256};
  case Domain::MMIO: return {256, 256};
  case Domain::VRAM: return {64, 192};
  case Domain::HBM:  return {64, 192};
  case Domain::CXL:  return {384, 512};
  case Domain::NVM:  return {1024, 4096};
  }
  return {256, 256};
}

// The seeded target constants (cost.py TargetProfile), read off the capability op.
struct Cap {
  SmallVector<int64_t> widths;  // sorted unique lane widths
  int64_t gatherPenalty = 0, cacheline = 64, memUnit = 1, baseOverhead = 4;
  int64_t thermalDensity = 64, powerDensity = 64, perOpHeat = 1, elemBytes = 4;
};

inline Cap readCap(TargetCapabilityOp capOp) {
  Cap h;
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
  return h;
}

inline int64_t bitLength(int64_t n) {  // ceil(log2 n) for HAM
  int64_t bits = 0;
  for (uint64_t v = static_cast<uint64_t>(n); v; v >>= 1)
    ++bits;
  return bits;
}

inline int opClassFor(StringRef op) {  // cost.py::_opclass via the op-string vocabulary
  if (op == "linalg.matmul")
    return 2;
  if (op.starts_with("histogram") || op.contains("scatter"))
    return 0;
  return 1;
}

inline int64_t stridePenalty(StrideClass sc, int strideK, const Cap &h) {
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

inline Cost baseCost(int64_t count, int streams, int opclass, int width,
                     int64_t penalty, TierQ8 tier, const Cap &h,
                     int64_t extraCompile = 0) {
  int64_t n = std::max<int64_t>(1, count);
  int64_t ceil = (n + width - 1) / width;
  Cost c{};
  c[0] = ceil * opclass;
  int64_t memShared = (n * streams * h.memUnit * tier.bw) >> 8;
  int64_t accessOps = ceil * streams;
  int64_t overhead = h.baseOverhead * penalty;
  c[1] = memShared + ((accessOps * overhead * tier.lat) >> 8);
  c[4] = extraCompile;
  c[5] = static_cast<int64_t>(width) * h.thermalDensity + ceil * h.perOpHeat;
  c[6] = static_cast<int64_t>(width) * h.powerDensity + ceil * h.perOpHeat;
  return c;
}

// realize._verify_cost -- the cost of discharging the claim's verify contract (the
// producer for the 12th cost axis). none/bounds are free (the bounds check is fused into
// the access the memory axis already prices); the expensive contracts carry a real,
// size-proportional, width-independent cost the optimizer can trade. Mirrors the oracle.
inline int64_t verifyCostFor(Verify v, int64_t count) {
  int64_t n = std::max<int64_t>(1, count);
  return (v == Verify::Exact || v == Verify::Hash) ? n : 0;
}

// realize.candidates_for -- the legal realizations (width + base cost), declared order.
inline SmallVector<Cand> candidatesFor(ClaimOp c, const Cap &h, Domain dom, bool ham) {
  StringRef op = c.getOp();
  StrideClass sc = c.getStrideClass();
  int64_t count = static_cast<int64_t>(c.getCount());
  int streams = static_cast<int>(c.getReads().size() + c.getWrites().size());
  int opclass = opClassFor(op);
  TierQ8 tier = tierForDomain(dom);
  int64_t gp = ham ? std::max<int64_t>(1, bitLength(count - 1)) : h.gatherPenalty;
  int64_t vcost = verifyCostFor(c.getVerify(), count);
  // Add the verification cost to every realization (width-independent, so the per-claim
  // selection is unchanged); applied on every return path.
  auto withVerify = [&](SmallVector<Cand> &o) -> SmallVector<Cand> & {
    if (vcost)
      for (Cand &cc : o)
        cc.cost[11] += vcost;
    return o;
  };

  SmallVector<Cand> out;
  if (op == "reduce.gather") {
    out.push_back({1, baseCost(count, streams, opclass, 1, 1, tier, h)});
    out.push_back({1, baseCost(count, streams, opclass, 1, gp, tier, h)});
    return withVerify(out);
  }
  switch (sc) {
  case StrideClass::Unit:
  case StrideClass::Scalar:
    for (int64_t w : h.widths)
      out.push_back({w, baseCost(count, streams, opclass, static_cast<int>(w), 1, tier, h)});
    break;
  case StrideClass::Strided:
    out.push_back({1, baseCost(count, streams, opclass, 1,
                               stridePenalty(sc, c.getStrideK(), h), tier, h)});
    out.push_back({1, baseCost(count, streams, opclass, 1, gp, tier, h)});
    break;
  case StrideClass::Cacheline: {
    int64_t uxw = std::min<int64_t>(8, h.widths.back());
    out.push_back({uxw, baseCost(count, streams, opclass, static_cast<int>(uxw), 2, tier, h,
                                 count / 4)});
    out.push_back({1, baseCost(count, streams, opclass, 1, gp, tier, h)});
    break;
  }
  case StrideClass::Random:
    out.push_back({1, baseCost(count, streams, opclass, 1, gp, tier, h)});
    break;
  case StrideClass::Tile:
    out.push_back({16, baseCost(count, streams, opclass, 16, 1, tier, h)});
    break;
  }
  return withVerify(out);
}

inline int64_t scalarize(const Cost &c, ArrayRef<int64_t> w) {
  int64_t s = 0;
  for (int i = 0; i < 12 && i < static_cast<int>(w.size()); ++i)
    s += c[i] * w[i];
  return s;
}

inline void applyFactor(Cost &c, const Factor &f) {
  for (int i = 0; i < 12; ++i)
    c[i] = (c[i] * f[i]) >> 8;
}

inline Factor deforestFactor() {  // producer->consumer: x0.75 memory
  Factor f = kIdentity;
  f[1] = 192;
  return f;
}
inline Factor cseFactor(int nrd, int nwr) {  // duplicate -> a copy (no recompute/reload)
  int64_t full = static_cast<int64_t>(nrd) + nwr, copy = 1 + nwr;
  Factor f = kIdentity;
  f[0] = 0;
  f[1] = (copy * 256) / std::max<int64_t>(1, full);
  return f;
}

// A claim's fused candidate column, in flat (phase, declared) order.
struct Column {
  ClaimOp claim;
  int32_t phase;
  SmallVector<StringRef> reads;
  SmallVector<StringRef> writes;
  SmallVector<Cand> cands;     // deforestation/CSE already baked in
  StringRef fusion;            // "", "deforest", or "cse"
};

inline void symRefs(ArrayAttr a, SmallVectorImpl<StringRef> &out) {
  for (Attribute x : a)
    if (auto r = dyn_cast<FlatSymbolRefAttr>(x))
      out.push_back(r.getValue());
}

// realize.fused_candidates: claims in (phase id, declared) order with the intra-phase
// deforestation/CSE credits baked into every candidate. The plan pass adds the
// path-based shared-input factor per edge on top of these.
inline std::vector<Column>
fusedColumns(Operation *root, const Cap &h,
             const llvm::DenseMap<StringRef, ResourceOp> &resByName) {
  llvm::DenseMap<StringRef, int32_t> phaseId;
  root->walk([&](PhaseOp p) { phaseId[p.getSymName()] = p.getId(); });
  SmallVector<ClaimOp> claims;
  root->walk([&](ClaimOp c) { claims.push_back(c); });
  llvm::stable_sort(claims, [&](ClaimOp a, ClaimOp z) {
    return phaseId.lookup(a.getPhase()) < phaseId.lookup(z.getPhase());
  });

  std::vector<Column> cols;
  int32_t curPhase = 0;
  bool havePhase = false;
  llvm::DenseSet<StringRef> produced;
  llvm::DenseMap<StringRef, int> version;
  llvm::StringSet<> seen;

  for (ClaimOp c : claims) {
    int32_t pid = phaseId.lookup(c.getPhase());
    if (!havePhase || pid != curPhase) {
      produced.clear();
      version.clear();
      seen.clear();
      curPhase = pid;
      havePhase = true;
    }
    Column col;
    col.claim = c;
    col.phase = pid;
    symRefs(c.getReads(), col.reads);
    symRefs(c.getWrites(), col.writes);
    SmallVector<StringRef> &writes = col.writes;

    Domain dom = c.getDomain();
    bool ham = false;
    if (!col.reads.empty())
      if (auto r = resByName.lookup(col.reads[0])) {
        dom = r.getDomainKind();
        ham = r.getAccess() && *r.getAccess() == Access::HAM;
      }
    col.cands = candidatesFor(c, h, dom, ham);

    std::string sig = c.getOp().str();
    for (StringRef rd : col.reads) {
      sig += "|";
      sig += rd.str();
      sig += "@";
      sig += std::to_string(version.lookup(rd));
    }
    bool cse = !col.reads.empty() && seen.count(sig);
    bool defo = false;
    if (!cse)
      for (StringRef rd : col.reads)
        if (produced.count(rd)) {
          defo = true;
          break;
        }
    if (cse || defo) {
      Factor f = cse ? cseFactor(static_cast<int>(col.reads.size()),
                                 static_cast<int>(writes.size()))
                     : deforestFactor();
      for (Cand &cc : col.cands)
        applyFactor(cc.cost, f);
      col.fusion = cse ? "cse" : "deforest";
    }
    if (!col.reads.empty())
      seen.insert(sig);
    for (StringRef w : writes)
      version[w] = version.lookup(w) + 1;
    for (StringRef w : writes)
      produced.insert(w);
    cols.push_back(std::move(col));
  }
  return cols;
}

// Like fusedColumns, but over an explicit claim list treated as a single parallel block (a
// compose `Leaf`): the same intra-block deforestation/CSE credits, no phase grouping. Used by
// -bcir-compose to price a region's leaf claims (realize.optimize of one phase).
inline std::vector<Column>
fusedColumnsFromClaims(ArrayRef<ClaimOp> claims, const Cap &h,
                       const llvm::DenseMap<StringRef, ResourceOp> &resByName) {
  std::vector<Column> cols;
  llvm::DenseSet<StringRef> produced;
  llvm::DenseMap<StringRef, int> version;
  llvm::StringSet<> seen;
  for (ClaimOp c : claims) {
    Column col;
    col.claim = c;
    col.phase = 0;
    symRefs(c.getReads(), col.reads);
    symRefs(c.getWrites(), col.writes);
    Domain dom = c.getDomain();
    bool ham = false;
    if (!col.reads.empty())
      if (auto r = resByName.lookup(col.reads[0])) {
        dom = r.getDomainKind();
        ham = r.getAccess() && *r.getAccess() == Access::HAM;
      }
    col.cands = candidatesFor(c, h, dom, ham);
    std::string sig = c.getOp().str();
    for (StringRef rd : col.reads) {
      sig += "|";
      sig += rd.str();
      sig += "@";
      sig += std::to_string(version.lookup(rd));
    }
    bool cse = !col.reads.empty() && seen.count(sig);
    bool defo = false;
    if (!cse)
      for (StringRef rd : col.reads)
        if (produced.count(rd)) {
          defo = true;
          break;
        }
    if (cse || defo) {
      Factor f = cse ? cseFactor(static_cast<int>(col.reads.size()),
                                 static_cast<int>(col.writes.size()))
                     : deforestFactor();
      for (Cand &cc : col.cands)
        applyFactor(cc.cost, f);
      col.fusion = cse ? "cse" : "deforest";
    }
    if (!col.reads.empty())
      seen.insert(sig);
    for (StringRef w : col.writes)
      version[w] = version.lookup(w) + 1;
    for (StringRef w : col.writes)
      produced.insert(w);
    cols.push_back(std::move(col));
  }
  return cols;
}

// --- the coupled min-plus shortest path (realize.optimize) ----------------------

inline bool sharesRead(ArrayRef<StringRef> a, ArrayRef<StringRef> b) {
  for (StringRef x : a)
    for (StringRef y : b)
      if (x == y)
        return true;
  return false;
}

// realize._context_factor (path-based half): a wide candidate whose wide predecessor
// shares a read operand discounts memory x0.75 (cache-line reuse); and, on a hot part
// (thetaThermal >= 60), a width>=16 candidate pays x1.25 heat/current (the AVX-512
// downclock). `thetaThermal` 0 == the cool regime (the coupling reduces to fusion).
inline Factor contextFactor(int64_t thetaThermal, ArrayRef<StringRef> prevReads,
                            int64_t prevWidth, ArrayRef<StringRef> curReads,
                            int64_t curWidth) {
  Factor f = kIdentity;
  if (prevWidth > 1 && curWidth > 1 && sharesRead(prevReads, curReads))
    f[1] = 192;
  if (thetaThermal >= 60 && curWidth >= 16) {
    f[5] = 320; // thermal x1.25
    f[6] = 320; // power x1.25
  }
  return f;
}

// The live thermal pressure of the first bcir.kbcir.theta op (0 == cool); the
// multiplicative thermal coupling depends on it (the emitter folds Theta into the
// policy weights separately, for scalarization).
inline int64_t firstThetaThermal(Operation *root) {
  int64_t thermal = 0;
  bool found = false;
  root->walk([&](Operation *op) {
    if (!found && op->getName().getStringRef() == "bcir.kbcir.theta") {
      if (auto a = op->getAttrOfType<IntegerAttr>("thermal"))
        thermal = a.getInt();
      found = true;
    }
  });
  return thermal;
}

// The layered tropical shortest path over the fused candidate columns. Returns the
// chosen candidate index per column (and the total score via `total`). Edge cost =
// scalarize(cand.cost coupled by the path-based context factor, w). Empty -> {}.
inline SmallVector<int> planChosen(const std::vector<Column> &cols,
                                   ArrayRef<int64_t> w, int64_t thetaThermal,
                                   int64_t &total) {
  total = 0;
  const int n = static_cast<int>(cols.size());
  if (n == 0 || w.empty())
    return {};
  const int64_t kInf = std::numeric_limits<int64_t>::max();
  std::vector<std::vector<int64_t>> dist(n);
  std::vector<std::vector<int>> back(n);
  for (int i = 0; i < n; ++i) {
    const auto &cands = cols[i].cands;
    dist[i].assign(cands.size(), kInf);
    back[i].assign(cands.size(), -1);
    for (size_t ci = 0; ci < cands.size(); ++ci) {
      if (i == 0) {
        // from SOURCE: no fusion predecessor, but the thermal coupling still applies.
        Cost e = cands[ci].cost;
        applyFactor(e, contextFactor(thetaThermal, {}, 0, cols[i].reads, cands[ci].width));
        dist[i][ci] = scalarize(e, w);
        continue;
      }
      for (size_t pi = 0; pi < cols[i - 1].cands.size(); ++pi) {
        if (dist[i - 1][pi] == kInf)
          continue;
        Factor f = contextFactor(thetaThermal, cols[i - 1].reads,
                                 cols[i - 1].cands[pi].width, cols[i].reads,
                                 cands[ci].width);
        Cost e = cands[ci].cost;
        applyFactor(e, f);
        int64_t d = dist[i - 1][pi] + scalarize(e, w);
        if (d < dist[i][ci]) {
          dist[i][ci] = d;
          back[i][ci] = static_cast<int>(pi);
        }
      }
    }
  }
  int lastBest = 0;
  for (size_t ci = 1; ci < cols[n - 1].cands.size(); ++ci)
    if (dist[n - 1][ci] < dist[n - 1][lastBest])
      lastBest = static_cast<int>(ci);
  total = dist[n - 1][lastBest];
  SmallVector<int> chosen(n, 0);
  chosen[n - 1] = lastBest;
  for (int i = n - 1; i > 0; --i)
    chosen[i - 1] = back[i][chosen[i]];
  return chosen;
}

// The first target.capability (the optimizer's H); null if none declared.
inline TargetCapabilityOp firstCapability(Operation *root) {
  TargetCapabilityOp cap;
  root->walk([&](TargetCapabilityOp t) {
    if (!cap)
      cap = t;
  });
  return cap;
}

// The first policy's weights (theta-folded by the emitter); empty if none.
inline ArrayRef<int64_t> firstWeights(Operation *root) {
  ArrayRef<int64_t> w;
  root->walk([&](KBCIRPolicyOp p) {
    if (w.empty())
      w = p.getWeights();
  });
  return w;
}

inline llvm::DenseMap<StringRef, ResourceOp> resourcesByName(Operation *root) {
  llvm::DenseMap<StringRef, ResourceOp> m;
  root->walk([&](ResourceOp r) { m[r.getSymName()] = r; });
  return m;
}

}  // namespace cm
}  // namespace bcir

#endif  // BCIR_LIB_PASSES_BCIRCOSTMODEL_H
