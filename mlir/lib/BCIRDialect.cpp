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
#include "mlir/IR/BuiltinTypes.h"
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

::mlir::LogicalResult GEMActivationOp::verify() {
  // The G1 plan record (mirrors bcir/kbcir/activation.py::check_activation -- the op-level shape/dtype
  // law gem.matmul validates inline; NO new globally-numbered R-law). The laws an activation must satisfy:
  //   (0) a known kind; (1) elementwise-shaped (spec shape's product == element count, all extents > 0);
  //   (2) a known, preserved dtype (f32|i32); (4) the QUARANTINE rule: a transcendental needs f32 (its
  //   libm edge returns float), relu may be i32 or f32; (5) the softmax axis divides the tensor and equals
  //   the last dim, the elementwise four declare axis_len 0; plus the roofline invariant bottleneck == max.
  ::llvm::StringRef kind = getKind();
  const bool isExact = (kind == "relu");
  const bool isTrans = (kind == "sigmoid" || kind == "tanh" || kind == "gelu" || kind == "softmax");
  if (!isExact && !isTrans)
    return emitOpError() << "unknown activation kind '" << kind
                         << "' (expected relu|sigmoid|tanh|gelu|softmax)";

  // (1) shape well-formed: a non-empty extent list with all positive dims; count = product.
  ::llvm::ArrayRef<int64_t> shape = getShape();
  if (shape.empty())
    return emitOpError() << kind << ": shape must be a non-empty (>=1-D) tensor extent";
  int64_t count = 1;
  for (int64_t d : shape) {
    if (d <= 0)
      return emitOpError() << kind << ": shape extents must be positive (got " << d << ")";
    count *= d;
  }

  // (2) dtype known + preserved (the op carries a single declared dtype: input == output == spec).
  ::llvm::StringRef dtype = getDtype();
  if (dtype != "f32" && dtype != "i32")
    return emitOpError() << kind << ": dtype '" << dtype << "' is not a known dtype (f32|i32)";

  // (4) the quarantine dtype rule: a transcendental routes a libm call (expf/tanhf) through the trusted
  // c.call.libm: edge, which returns float -- so it needs an f32 result, never i32. relu (the exact,
  // 0-ULP integer/Q-fixed-clean activation) may be i32 OR f32.
  if (isTrans && dtype != "f32")
    return emitOpError() << kind << ": transcendental activation needs f32 (the libm edge returns float), "
                         << "got '" << dtype << "'";

  // (5) the softmax axis: a valid last-axis length that divides the tensor and equals the last dim; the
  // elementwise four declare axis_len 0 (no reduction axis).
  int64_t axisLen = static_cast<int64_t>(getAxisLen());
  if (kind == "softmax") {
    int64_t last = shape.back();
    if (axisLen != last)
      return emitOpError() << "softmax: axis_len " << axisLen << " != last shape dim " << last;
    if (axisLen < 1 || (count % axisLen != 0))
      return emitOpError() << "softmax: axis_len " << axisLen << " must be >=1 and divide count " << count;
  } else if (axisLen != 0) {
    return emitOpError() << kind << ": elementwise activation must declare axis_len 0, got " << axisLen;
  }

  // width in [1, count]: a lane wider than the tensor is pointless (mirrors _divisor_widths' cap).
  int64_t width = static_cast<int64_t>(getWidth());
  if (width < 1 || width > count)
    return emitOpError() << kind << ": width " << width << " must be in [1, count " << count << "]";

  // the roofline invariant: bottleneck == max(compute_cost, mem_cost), the max,+ step the dual-semiring
  // width search minimizes (identical to gem.matmul's bottleneck check).
  int64_t compute = static_cast<int64_t>(getComputeCost()),
          mem = static_cast<int64_t>(getMemCost()), bn = static_cast<int64_t>(getBottleneck());
  if (compute < 0 || mem < 0)
    return emitOpError() << kind << ": compute_cost and mem_cost must be non-negative";
  int64_t mx = compute > mem ? compute : mem;
  if (bn != mx)
    return emitOpError() << kind << ": bottleneck must equal max(compute_cost, mem_cost) = " << mx
                         << " (got " << bn << ")";

  // the R17 accuracy axis: an EXACT activation (relu) carries 0 ULP from its own op, so its only accuracy
  // cost is the Q8 bridge step itself when quantized -- identical to a transcendental, whose libm kernel is
  // trusted/exact. So acc_bound must be 0 when dense (quant_bits == 0) and the 1-ULP bridge bound when
  // quantized (quant_bits > 0). (quantization_error_bound() == 1 ULP; mirrors activation.cost_vector.)
  int64_t accBound = static_cast<int64_t>(getAccBound());
  int64_t quantBits = static_cast<int64_t>(getQuantBits());
  int64_t wantAcc = quantBits > 0 ? 1 : 0;
  if (accBound != wantAcc)
    return emitOpError() << kind << ": acc_bound must be " << wantAcc << " (the R17 Q8-bridge bound, "
                         << (quantBits > 0 ? "1 ULP for a quantized realization" : "0 for a dense one")
                         << "), got " << accBound;
  return ::mlir::success();
}

::mlir::LogicalResult GEMMatmulBufferOp::verify() {
  // Buffer-form matmul: real memref operands C += A*B with a tile plan. Require A/B/C to be
  // rank-2 f32 memrefs with STATIC shapes, the contraction dims to agree (A.dim1==B.dim0 = K,
  // A.dim0==C.dim0 = M, B.dim1==C.dim1 = N), each tile extent in [1, its dim], and a known
  // loop order. M/N/K are derived from the static shapes, not duplicated as attributes.
  auto checkMemref = [&](::mlir::Value v, const char *name)
      -> ::mlir::FailureOr<::mlir::MemRefType> {
    auto mr = ::mlir::dyn_cast<::mlir::MemRefType>(v.getType());
    if (!mr)
      return emitOpError() << name << " must be a memref (got " << v.getType() << ")";
    if (mr.getRank() != 2)
      return emitOpError() << name << " must be rank-2 (got rank " << mr.getRank() << ")";
    if (!mr.getElementType().isF32())
      return emitOpError() << name << " element type must be f32 (got " << mr.getElementType()
                           << ")";
    if (!mr.hasStaticShape())
      return emitOpError() << name << " must have a static shape (got " << mr << ")";
    return mr;
  };
  auto a = checkMemref(getA(), "A");
  if (::mlir::failed(a))
    return ::mlir::failure();
  auto b = checkMemref(getB(), "B");
  if (::mlir::failed(b))
    return ::mlir::failure();
  auto c = checkMemref(getC(), "C");
  if (::mlir::failed(c))
    return ::mlir::failure();

  int64_t M = a->getDimSize(0), K = a->getDimSize(1);
  int64_t Kb = b->getDimSize(0), N = b->getDimSize(1);
  int64_t Mc = c->getDimSize(0), Nc = c->getDimSize(1);
  if (K != Kb)
    return emitOpError() << "contraction dim mismatch: A is " << M << "x" << K << ", B is " << Kb
                         << "x" << N << " (A.dim1 must equal B.dim0)";
  if (M != Mc)
    return emitOpError() << "M mismatch: A.dim0 = " << M << " but C.dim0 = " << Mc;
  if (N != Nc)
    return emitOpError() << "N mismatch: B.dim1 = " << N << " but C.dim1 = " << Nc;

  int64_t tm = static_cast<int64_t>(getTileM()), tn = static_cast<int64_t>(getTileN()),
          tk = static_cast<int64_t>(getTileK());
  if (tm < 1 || tm > M || tn < 1 || tn > N || tk < 1 || tk > K)
    return emitOpError() << "each tile extent must be in [1, dim] (tiles " << tm << "x" << tn << "x"
                         << tk << " vs dims " << M << "x" << N << "x" << K << ")";
  ::llvm::StringRef lo = getLoopOrder();
  if (lo != "ijk" && lo != "ikj" && lo != "jik")
    return emitOpError() << "loop_order must be one of ijk|ikj|jik (got '" << lo << "')";
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
