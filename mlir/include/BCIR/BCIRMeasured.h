//===- BCIRMeasured.h - the measured alternatives' one spelling -----*- C++ -*-===//
//
// PX-3b-2: the law rail's twin of bcir/kbcir/cost.py `measured_alternative_key`,
// `spelled_as_opcode` and `MEASURED_PER_KEY_MAX`, and of the control-op set `realize._offer_rows`
// never measures -- shared by `bcir.target.capability`'s
// verifier (BCIRDialect.cpp) and the cost model (BCIRCostModel.h), so the spelling the verifier
// admits is the one the offer prices.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_BCIRMEASURED_H
#define BCIR_BCIRMEASURED_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <utility>

namespace bcir {

// (0, bit) for an instance `i<bit>` (1, 2, 4 or 8), (1, k) for a worker count `t<k>` (1 ..
// 32767) -- the order a measured table holds its alternatives in -- and (-1, 0) for any other
// spelling: ASCII digits only, no leading zero, at most five of them.
inline std::pair<int, int64_t> measuredAlternativeKey(llvm::StringRef s) {
  const std::pair<int, int64_t> none{-1, 0};
  if (s.size() < 2 || (s[0] != 'i' && s[0] != 't'))
    return none;
  llvm::StringRef digits = s.drop_front();
  if (digits.size() > 5 || digits[0] == '0')
    return none;
  int64_t v = 0;
  for (char ch : digits) {
    if (ch < '0' || ch > '9')
      return none;
    v = v * 10 + (ch - '0');
  }
  if (s[0] == 'i')
    return (v == 1 || v == 2 || v == 4 || v == 8) ? std::pair<int, int64_t>{0, v} : none;
  return v <= 32767 ? std::pair<int, int64_t>{1, v} : none;
}

// cost.py spelled_as_opcode: an ASCII capital, then ASCII capitals, digits and underscores -- the
// emitter's spelling of an op-less claim's `op` (its opcode's name), so a measured entry spelled
// so would name op-less claims the oracle never offers it; no measured table holds one.
inline bool isOpcodeSpelled(llvm::StringRef op) {
  if (op.empty() || op[0] < 'A' || op[0] > 'Z')
    return false;
  for (char ch : op.drop_front())
    if (!((ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_'))
      return false;
  return true;
}

// The most measured entries one (op, count) holds (cost.py MEASURED_PER_KEY_MAX).
constexpr int kMeasuredPerKeyMax = 7;

// realize._CONTROL_NOOPS and BARRIER (model/opcodes.py: NOP 0, BARRIER 10, PHASE_ENTER 11,
// PHASE_LEAVE 12, PROV_NOTE 17): a control op is never offered a measured alternative.
inline bool isMeasuredControlOpcode(int64_t opcode) {
  return opcode == 0 || opcode == 10 || opcode == 11 || opcode == 12 || opcode == 17;
}

} // namespace bcir

#endif // BCIR_BCIRMEASURED_H
