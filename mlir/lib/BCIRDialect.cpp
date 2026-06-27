//===- BCIRDialect.cpp - BCIR dialect registration ------------------------===//
//
// Registers the BCIR dialect family (types, attributes, enums, ops) generated
// from the .td law. Built by the out-of-tree CMake (see ../CMakeLists.txt) into
// the `bcir-opt` tool, which parses/verifies the pretty ODS corpus.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"

#include "BCIR/BCIRAttrs.h"
#include "BCIR/BCIROps.h"
#include "BCIR/BCIRTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace bcir;

#include "BCIRDialect.cpp.inc"

#include "BCIREnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "BCIRAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "BCIRTypes.cpp.inc"

#define GET_OP_CLASSES
#include "BCIROps.cpp.inc"

// --- op verifiers (hasVerifier=1) ------------------------------------------------
// Parse-time structural well-formedness, run on every parse/builder. Distinct from
// -bcir-verify, which carries the cross-op semantic R-laws (RID resolution, the phase
// DAG, plan legality, ...). These reject a malformed op at the point it is built.
::mlir::LogicalResult ResourceOp::verify() {
  int64_t align = getAlign();
  if (align <= 0 || (align & (align - 1)) != 0)
    return emitOpError() << "align must be a positive power of two (got " << align << ")";
  for (int64_t d : getShape())
    if (d <= 0)
      return emitOpError() << "shape extents must be positive (got " << d << ")";
  return ::mlir::success();
}

::mlir::LogicalResult GEMLaneSegmentOp::verify() {
  int64_t width = getWidth();
  if (width <= 0 || (width & (width - 1)) != 0)
    return emitOpError() << "width must be a positive power of two (got " << width << ")";
  if (getStrideK() <= 0)
    return emitOpError() << "stride_k must be positive (got " << getStrideK() << ")";
  return ::mlir::success();
}

::mlir::LogicalResult GEMMatmulOp::verify() {
  // The B1 plan record: positive shape, each tile in [1, dim], a known loop order, and the
  // bottleneck = max(compute, memory) (the max,+ roofline the dual-semiring search minimizes).
  int64_t m = static_cast<int64_t>(getM()), n = static_cast<int64_t>(getN()),
          k = static_cast<int64_t>(getK());
  if (m <= 0 || n <= 0 || k <= 0)
    return emitOpError() << "matmul dims m/n/k must be positive (got " << m << "x" << n << "x" << k
                         << ")";
  int64_t tm = static_cast<int64_t>(getTileM()), tn = static_cast<int64_t>(getTileN()),
          tk = static_cast<int64_t>(getTileK());
  if (tm < 1 || tm > m || tn < 1 || tn > n || tk < 1 || tk > k)
    return emitOpError() << "each tile extent must be in [1, dim] (tiles " << tm << "x" << tn << "x"
                         << tk << " vs dims " << m << "x" << n << "x" << k << ")";
  ::llvm::StringRef lo = getLoopOrder();
  if (lo != "ijk" && lo != "ikj" && lo != "jik")
    return emitOpError() << "loop_order must be one of ijk|ikj|jik (got '" << lo << "')";
  int64_t compute = static_cast<int64_t>(getComputeCost()),
          mem = static_cast<int64_t>(getMemCost()), bn = static_cast<int64_t>(getBottleneck());
  if (compute < 0 || mem < 0)
    return emitOpError() << "compute_cost and mem_cost must be non-negative";
  int64_t mx = compute > mem ? compute : mem;
  if (bn != mx)
    return emitOpError() << "bottleneck must equal max(compute_cost, mem_cost) = " << mx << " (got "
                         << bn << ")";
  // quant_bits is an unsigned i32 attr (0 = dense), so it cannot be negative -- no further check.
  return ::mlir::success();
}

::mlir::LogicalResult ClaimOp::verify() {
  // getCount() is uint64_t (an i64 attr); reinterpret signed so a negative literal is caught.
  int64_t count = static_cast<int64_t>(getCount());
  if (count < 0)
    return emitOpError() << "count must be non-negative (got " << count << ")";
  if (getStrideK() == 0)
    return emitOpError() << "stride_k must be positive (got " << getStrideK() << ")";
  return ::mlir::success();
}

::mlir::LogicalResult TargetCapabilityOp::verify() {
  int64_t cl = getCacheline();
  if (cl <= 0 || (cl & (cl - 1)) != 0)
    return emitOpError() << "cacheline must be a positive power of two (got " << cl << ")";
  for (int64_t w : getLaneWidths())
    if (w <= 0)
      return emitOpError() << "lane widths must be positive (got " << w << ")";
  return ::mlir::success();
}

void BCIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "BCIROps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "BCIRAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "BCIRTypes.cpp.inc"
      >();
}
