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
#include "llvm/ADT/StringSwitch.h"

#include <array>
#include <optional>

using namespace mlir;

namespace bcir {

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
  auto intersects = [](const llvm::DenseSet<StringRef> &x,
                       const llvm::DenseSet<StringRef> &y) {
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
  out[0] = cv.getCompute();    out[1] = cv.getMemory();
  out[2] = cv.getFabric();     out[3] = cv.getSync();
  out[4] = cv.getCompile();    out[5] = cv.getThermal();
  out[6] = cv.getPower();      out[7] = cv.getReliability();
  out[8] = cv.getSecurity();   out[9] = cv.getAccuracy();
  out[10] = cv.getContention(); out[11] = cv.getVerification();
}

// score(pi) = sum_d w_d * cost_d -- the scalarized K_BCIR cost (the dot product
// the min-plus selection minimizes; LangRef Sec. 2 degenerate rail).
inline int64_t scalarize(CostVectorAttr cv, ArrayRef<int64_t> w) {
  int64_t cost[12];
  costToArray(cv, cost);
  int64_t s = 0;
  for (int i = 0; i < 12 && i < static_cast<int>(w.size()); ++i)
    s += cost[i] * w[i];
  return s;
}

// The mnemonic for a lane (for annotation; matches BCIRAttrs.td spellings).
inline StringRef laneSpelling(Lane l) {
  switch (l) {
  case Lane::U:   return "u";
  case Lane::UX:  return "ux";
  case Lane::T:   return "t";
  case Lane::GGG: return "ggg";
  case Lane::A:   return "a";
  case Lane::H:   return "h";
  }
  return "?";
}

// The canonical (preferred) lane the classifier assigns to an access pattern --
// the most specific legal lane for the stride class (mirrors the oracle's lane
// classification: unit/strided stream on U, cacheline on UX, tile on T, random
// quarantined to GGG).
inline Lane canonicalLane(StrideClass sc) {
  switch (sc) {
  case StrideClass::Scalar:    return Lane::U;
  case StrideClass::Unit:      return Lane::U;
  case StrideClass::Strided:   return Lane::U;
  case StrideClass::Cacheline: return Lane::UX;
  case StrideClass::Tile:      return Lane::T;
  case StrideClass::Random:    return Lane::GGG;
  }
  return Lane::U;
}

}  // namespace bcir

#endif  // BCIR_LIB_PASSES_BCIRPASSSUPPORT_H
