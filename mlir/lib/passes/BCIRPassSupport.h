//===- BCIRPassSupport.h - shared helpers for the BCIR passes ----*- C++ -*-===//
//
// Internal (lib-private) helpers shared by the BCIR pass translation units: lane/
// hazard legality, cost-vector access + scalarization, lane spellings. Header-only
// (inline) so each modular pass TU links one definition. NOT a public API.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_LIB_PASSES_BCIRPASSSUPPORT_H
#define BCIR_LIB_PASSES_BCIRPASSSUPPORT_H

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>

using namespace mlir;

namespace bcir {

// Integer attributes originate in parsed IR.  Never let malformed geometry or
// cost metadata invoke signed-overflow UB inside a verifier/pass: callers can
// either reject the operation or deliberately saturate an informs-only score.
inline bool checkedAddNonnegative(int64_t a, int64_t b, int64_t &out) {
  if (a < 0 || b < 0 || b > std::numeric_limits<int64_t>::max() - a)
    return false;
  out = a + b;
  return true;
}

inline bool checkedMulNonnegative(int64_t a, int64_t b, int64_t &out) {
  if (a < 0 || b < 0 || (a != 0 && b > std::numeric_limits<int64_t>::max() / a))
    return false;
  out = a * b;
  return true;
}

inline bool checkedMul3Nonnegative(int64_t a, int64_t b, int64_t c, int64_t &out) {
  int64_t first;
  return checkedMulNonnegative(a, b, first) && checkedMulNonnegative(first, c, out);
}

inline int64_t saturatingAddNonnegative(int64_t a, int64_t b) {
  int64_t out;
  return checkedAddNonnegative(a, b, out) ? out : std::numeric_limits<int64_t>::max();
}

inline int64_t saturatingMulNonnegative(int64_t a, int64_t b) {
  int64_t out;
  return checkedMulNonnegative(a, b, out) ? out : std::numeric_limits<int64_t>::max();
}

inline int64_t saturatingAddSigned(int64_t a, int64_t b) {
  int64_t out;
  if (!llvm::AddOverflow(a, b, out))
    return out;
  return b < 0 ? std::numeric_limits<int64_t>::min() : std::numeric_limits<int64_t>::max();
}

inline int64_t saturatingMulSigned(int64_t a, int64_t b) {
  int64_t out;
  if (!llvm::MulOverflow(a, b, out))
    return out;
  return (a < 0) != (b < 0) ? std::numeric_limits<int64_t>::min()
                            : std::numeric_limits<int64_t>::max();
}

// Materializing one operation per tile/stripe is intentionally bounded.  The
// compact lowering is not a license for parsed dimensions to allocate millions
// of IR nodes; larger workloads must use a loop-form lowering.
inline constexpr int64_t kMaxMaterializedBlocksPerOp = 65536;

inline bool ceilDivPositive(int64_t numerator, int64_t denominator, int64_t &out) {
  if (numerator <= 0 || denominator <= 0)
    return false;
  out = numerator / denominator + (numerator % denominator != 0);
  return true;
}

inline bool checkedTileCount(int64_t m, int64_t n, int64_t k, int64_t tm, int64_t tn, int64_t tk,
                             int64_t &out) {
  int64_t ni, nj, nk;
  return ceilDivPositive(m, tm, ni) && ceilDivPositive(n, tn, nj) && ceilDivPositive(k, tk, nk) &&
         checkedMul3Nonnegative(ni, nj, nk, out);
}

// Shared checked form of the GEM roofline's derived traffic.  The op verifier
// bounds M*N*K, but C is read once per K tile plus one final write, so traffic
// can still exceed the signed-i64 cost domain and must be rejected explicitly.
inline bool checkedGemmDerived(int64_t m, int64_t n, int64_t k, int64_t tm, int64_t tn, int64_t tk,
                               int64_t &macs, int64_t &traffic, int64_t &workingSet) {
  int64_t nTiles, mTiles, kTiles;
  int64_t aReads, bReads, cTraffic, mn, kPasses;
  int64_t wsA, wsB, wsC, wsAB;
  int64_t abTraffic;
  return ceilDivPositive(n, tn, nTiles) && ceilDivPositive(m, tm, mTiles) &&
         ceilDivPositive(k, tk, kTiles) && checkedMul3Nonnegative(m, n, k, macs) &&
         checkedMul3Nonnegative(m, k, nTiles, aReads) &&
         checkedMul3Nonnegative(k, n, mTiles, bReads) && checkedMulNonnegative(m, n, mn) &&
         checkedAddNonnegative(kTiles, 1, kPasses) &&
         checkedMulNonnegative(mn, kPasses, cTraffic) &&
         checkedAddNonnegative(aReads, bReads, abTraffic) &&
         checkedAddNonnegative(abTraffic, cTraffic, traffic) &&
         checkedMulNonnegative(tm, tk, wsA) && checkedMulNonnegative(tk, tn, wsB) &&
         checkedMulNonnegative(tm, tn, wsC) && checkedAddNonnegative(wsA, wsB, wsAB) &&
         checkedAddNonnegative(wsAB, wsC, workingSet);
}

// Which lanes are legal for each access-pattern shape (LangRef R6).
inline bool laneLegalForStride(Lane lane, StrideClass sc) {
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

// A hazard contract that orders accesses (R5: atomic semantics demand one).
inline bool hazardOrdered(HazardMode h) {
  return h == HazardMode::Atomic || h == HazardMode::Barriered;
}

// A claim on the decoupled GGG/random tail (CT2): same-phase conflicts through
// it lose their implicit wave serialization (R5).
inline bool claimSparse(ClaimOp c) {
  return c.getLane() == Lane::GGG || c.getStrideClass() == StrideClass::Random;
}

inline llvm::DenseSet<StringRef> symbolSet(ArrayAttr refs) {
  llvm::DenseSet<StringRef> out;
  for (Attribute a : refs)
    if (auto ref = dyn_cast<FlatSymbolRefAttr>(a))
      out.insert(ref.getValue());
  return out;
}

// Look up a named dimension of a 12-d K_BCIR cost vector (R8/R9 budget law).
inline std::optional<int64_t> costDim(CostVectorAttr cv, StringRef name) {
  return llvm::StringSwitch<std::optional<int64_t>>(name)
      .Case("compute", cv.getCompute())
      .Case("memory", cv.getMemory())
      .Case("fabric", cv.getFabric())
      .Case("sync", cv.getSync())
      .Case("compile", cv.getCompile())
      .Case("thermal", cv.getThermal())
      .Case("power", cv.getPower())
      .Case("reliability", cv.getReliability())
      .Case("security", cv.getSecurity())
      .Case("accuracy", cv.getAccuracy())
      .Case("contention", cv.getContention())
      .Case("verification", cv.getVerification())
      .Default(std::nullopt);
}

// A read/write hazard between two claims (RAW / WAR / WAW).
inline bool claimsConflict(ClaimOp a, ClaimOp b) {
  auto aw = symbolSet(a.getWrites()), ar = symbolSet(a.getReads());
  auto bw = symbolSet(b.getWrites()), br = symbolSet(b.getReads());
  auto intersects = [](const llvm::DenseSet<StringRef> &x, const llvm::DenseSet<StringRef> &y) {
    for (StringRef s : x)
      if (y.count(s))
        return true;
    return false;
  };
  return intersects(aw, br) || intersects(aw, bw) || intersects(bw, ar);
}

// The 12 cost-vector dimensions in canonical (DIMS) order -- the order the
// K_BCIR policy weight vector indexes (mirrors bcir/kbcir/cost.py DIMS).
inline void costToArray(CostVectorAttr cv, int64_t out[12]) {
  out[0] = cv.getCompute();
  out[1] = cv.getMemory();
  out[2] = cv.getFabric();
  out[3] = cv.getSync();
  out[4] = cv.getCompile();
  out[5] = cv.getThermal();
  out[6] = cv.getPower();
  out[7] = cv.getReliability();
  out[8] = cv.getSecurity();
  out[9] = cv.getAccuracy();
  out[10] = cv.getContention();
  out[11] = cv.getVerification();
}

// score(pi) = sum_d w_d * cost_d -- the scalarized K_BCIR cost (the dot product
// the min-plus selection minimizes; LangRef Sec. 2 degenerate rail).
inline int64_t scalarize(CostVectorAttr cv, ArrayRef<int64_t> w) {
  int64_t cost[12];
  costToArray(cv, cost);
  int64_t s = 0;
  for (int i = 0; i < 12 && i < static_cast<int>(w.size()); ++i) {
    // Negative cost/weight metadata is invalid.  Standalone optimization
    // passes treat it as maximally bad even if -bcir-verify was not run first.
    if (cost[i] < 0 || w[i] < 0)
      return std::numeric_limits<int64_t>::max();
    s = saturatingAddNonnegative(s, saturatingMulNonnegative(cost[i], w[i]));
  }
  return s;
}

// The mnemonic for a lane (for annotation; matches BCIRAttrs.td spellings).
inline StringRef laneSpelling(Lane l) {
  switch (l) {
  case Lane::U:
    return "u";
  case Lane::UX:
    return "ux";
  case Lane::T:
    return "t";
  case Lane::GGG:
    return "ggg";
  case Lane::A:
    return "a";
  case Lane::H:
    return "h";
  }
  return "?";
}

// The canonical (preferred) lane the classifier assigns to an access pattern --
// the most specific legal lane for the stride class (mirrors the oracle's lane
// classification: unit/strided stream on U, cacheline on UX, tile on T, random
// quarantined to GGG).
inline Lane canonicalLane(StrideClass sc) {
  switch (sc) {
  case StrideClass::Scalar:
    return Lane::U;
  case StrideClass::Unit:
    return Lane::U;
  case StrideClass::Strided:
    return Lane::U;
  case StrideClass::Cacheline:
    return Lane::UX;
  case StrideClass::Tile:
    return Lane::T;
  case StrideClass::Random:
    return Lane::GGG;
  }
  return Lane::U;
}

// --- module scope (S0-5) --------------------------------------------------------------------
//
// A `bcir.module` is a symbol table (IsolatedFromAbove) and the unit the laws quantify
// over: registries, phases, paths, plans, policies and capabilities are ITS symbols. A
// pass that builds its maps with `root->walk(...)` from whatever root it was handed lets
// one module's claim resolve another module's resource, two modules trip R1 against each
// other's RIDs, and a selection price a namesake path from the wrong module -- the
// "MLIR module scope violated" finding of the 2026-07/08 assessment. These two helpers are
// the ONE predicate every scoped pass shares (laws.md L14): the scopes of a root, and a
// walk that never leaves its scope.

inline bool isModuleScope(Operation *op) {
  return op->getName().getStringRef() == "bcir.module";
}

// Every `bcir.module` at or under `root` is a scope; when the root is not itself a module
// it is the OUTER scope too, holding whatever lies outside any module -- so a module-free
// file (the atomic, ASN.1 and artifact fixtures) is one scope and verifies exactly as it
// always did. Modules first, in walk order, then the outer scope.
template <typename Fn>
void forEachScope(Operation *root, Fn &&fn) {
  SmallVector<Operation *> scopes;
  root->walk([&](Operation *op) {
    if (isModuleScope(op))
      scopes.push_back(op);
  });
  if (!isModuleScope(root))
    scopes.push_back(root);
  for (Operation *scope : scopes)
    fn(scope);
}

namespace detail {
template <typename OpT>
void collectScope(Operation *op, SmallVectorImpl<OpT> &out) {
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &child : block) {
        if (isModuleScope(&child))
          continue; // another module is its own scope, never part of this one
        collectScope<OpT>(&child, out);
      }
  if (auto typed = dyn_cast<OpT>(op))
    out.push_back(typed);
}
} // namespace detail

// `walkScope(scope, [&](SomeOp op) { ... })`: Operation::walk restricted to the scope --
// post-order like the original, the scope itself included, nested modules skipped. The
// matches are collected before the callback runs, so a callback that annotates or emits
// diagnostics never perturbs the traversal.
template <typename Fn>
void walkScope(Operation *scope, Fn &&fn) {
  using OpT = std::decay_t<typename llvm::function_traits<std::decay_t<Fn>>::template arg_t<0>>;
  SmallVector<OpT> found;
  detail::collectScope<OpT>(scope, found);
  for (OpT op : found)
    fn(op);
}

} // namespace bcir

#endif // BCIR_LIB_PASSES_BCIRPASSSUPPORT_H
