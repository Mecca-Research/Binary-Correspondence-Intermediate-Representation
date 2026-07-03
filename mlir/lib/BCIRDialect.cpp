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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <numeric>

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

::mlir::LogicalResult GEMConvOp::verify() {
  // The G7 plan record (mirrors bcir/kbcir/conv.py::check_conv + the realization/cost invariants -- the
  // op-level shape/dtype/realization law gem.matmul/gem.activation validate inline; NO new globally-
  // numbered R-law). A 2-D single-group conv is a STRUCTURED MATMUL, priced through the matmul roofline.
  int64_t inC = static_cast<int64_t>(getInC()), inH = static_cast<int64_t>(getInH()),
          inW = static_cast<int64_t>(getInW());
  int64_t outC = static_cast<int64_t>(getOutC()), kh = static_cast<int64_t>(getKh()),
          kw = static_cast<int64_t>(getKw());
  int64_t stride = static_cast<int64_t>(getStride()), pad = static_cast<int64_t>(getPad());

  // (4) hyperparameter sanity (checked first; the derived out dims below assume stride >= 1).
  if (stride < 1)
    return emitOpError() << "conv: stride must be >= 1 (got " << stride << ")";
  if (pad < 0)
    return emitOpError() << "conv: pad must be >= 0 (got " << pad << ")";
  // (1) positive extents.
  if (inC < 1 || inH < 1 || inW < 1 || outC < 1 || kh < 1 || kw < 1)
    return emitOpError() << "conv: in_c/in_h/in_w/out_c/kh/kw must all be >= 1 (got " << inC << "x"
                         << inH << "x" << inW << ", out_c " << outC << ", kernel " << kh << "x" << kw
                         << ")";
  // (1b) the kernel fits the padded input frame.
  if (kh > inH + 2 * pad || kw > inW + 2 * pad)
    return emitOpError() << "conv: kernel " << kh << "x" << kw << " does not fit the padded input "
                         << (inH + 2 * pad) << "x" << (inW + 2 * pad);

  // (3) the DERIVED output dims: out = floor((in + 2*pad - k)/stride) + 1 (the standard valid-after-pad
  // formula); the op's declared out_h/out_w must equal them.
  int64_t outH = static_cast<int64_t>(getOutH()), outW = static_cast<int64_t>(getOutW());
  int64_t wantH = (inH + 2 * pad - kh) / stride + 1;
  int64_t wantW = (inW + 2 * pad - kw) / stride + 1;
  if (wantH < 1 || wantW < 1)
    return emitOpError() << "conv: output is empty (derived out_h=" << wantH << ", out_w=" << wantW
                         << ")";
  if (outH != wantH || outW != wantW)
    return emitOpError() << "conv: declared out " << outH << "x" << outW
                         << " != the derived floor((in+2*pad-k)/stride)+1 = " << wantH << "x" << wantW;

  // (3b) the equivalent im2col gemm dims: gemm_m = out_h*out_w (the output pixels), gemm_n = out_c,
  // gemm_k = in_c*kh*kw (the taps / the reduction). This is what makes a conv a STRUCTURED MATMUL.
  int64_t gm = static_cast<int64_t>(getGemmM()), gn = static_cast<int64_t>(getGemmN()),
          gk = static_cast<int64_t>(getGemmK());
  int64_t wantM = outH * outW, wantN = outC, wantK = inC * kh * kw;
  if (gm != wantM || gn != wantN || gk != wantK)
    return emitOpError() << "conv: declared im2col gemm dims (" << gm << "," << gn << "," << gk
                         << ") != (out_h*out_w, out_c, in_c*kh*kw) = (" << wantM << "," << wantN << ","
                         << wantK << ")";

  // (2) dtype known + preserved (the op carries a single declared dtype: input == output == spec).
  ::llvm::StringRef dtype = getDtype();
  if (dtype != "f32" && dtype != "i32")
    return emitOpError() << "conv: dtype '" << dtype << "' is not a known dtype (f32|i32)";

  // the realization: a known strategy (direct|im2col), each inner gemm tile in [1, gemm dim], and a known
  // loop order. The strategy is the K_BCIR-chosen lowering (mirrors conv.STRATEGIES); for the direct
  // stream the tile is the WHOLE gemm (untiled), for im2col it is the chosen matmul tile.
  ::llvm::StringRef strat = getStrategy();
  if (strat != "direct" && strat != "im2col")
    return emitOpError() << "conv: strategy must be one of direct|im2col (got '" << strat << "')";
  int64_t tm = static_cast<int64_t>(getTileM()), tn = static_cast<int64_t>(getTileN()),
          tk = static_cast<int64_t>(getTileK());
  if (tm < 1 || tm > gm || tn < 1 || tn > gn || tk < 1 || tk > gk)
    return emitOpError() << "conv: each inner gemm tile must be in [1, gemm dim] (tiles " << tm << "x"
                         << tn << "x" << tk << " vs gemm dims " << gm << "x" << gn << "x" << gk << ")";
  // the direct stream is the UNTILED gemm (the whole M,N,K) -- it can not block the reduction the way the
  // im2col gemm tile does (mirrors conv.cost_of's direct branch == matmul.cost_of at tile = whole dims).
  if (strat == "direct" && (tm != gm || tn != gn || tk != gk))
    return emitOpError() << "conv: the direct strategy is the UNTILED gemm -- tile must be the whole "
                         << gm << "x" << gn << "x" << gk << " (got " << tm << "x" << tn << "x" << tk
                         << ")";
  ::llvm::StringRef lo = getLoopOrder();
  if (lo != "ijk" && lo != "ikj" && lo != "jik")
    return emitOpError() << "conv: loop_order must be one of ijk|ikj|jik (got '" << lo << "')";

  // the roofline invariant: bottleneck == max(compute_cost, mem_cost), the max,+ step the dual-semiring
  // strategy search minimizes (identical to gem.matmul's bottleneck check).
  int64_t compute = static_cast<int64_t>(getComputeCost()),
          mem = static_cast<int64_t>(getMemCost()), bn = static_cast<int64_t>(getBottleneck());
  if (compute < 0 || mem < 0)
    return emitOpError() << "conv: compute_cost and mem_cost must be non-negative";
  int64_t mx = compute > mem ? compute : mem;
  if (bn != mx)
    return emitOpError() << "conv: bottleneck must equal max(compute_cost, mem_cost) = " << mx
                         << " (got " << bn << ")";

  // the R17 accuracy axis: a conv's MACs are exact, so its only certified error is the Q8 bridge step when
  // quantized -- acc_bound must be 0 dense (quant_bits == 0) / the 1-ULP bridge bound quantized
  // (quant_bits > 0). (quantization_error_bound() == 1 ULP; mirrors conv.cost_vector.)
  int64_t accBound = static_cast<int64_t>(getAccBound());
  int64_t quantBits = static_cast<int64_t>(getQuantBits());
  int64_t wantAcc = quantBits > 0 ? 1 : 0;
  if (accBound != wantAcc)
    return emitOpError() << "conv: acc_bound must be " << wantAcc << " (the R17 Q8-bridge bound, "
                         << (quantBits > 0 ? "1 ULP for a quantized realization" : "0 for a dense one")
                         << "), got " << accBound;
  return ::mlir::success();
}

::mlir::LogicalResult GEMAttentionOp::verify() {
  // The G7 attention plan record (mirrors bcir/kbcir/attention.py::check_attention + the realization/cost
  // invariants -- the op-level shape/dtype/realization law gem.matmul/gem.activation/gem.conv validate
  // inline; NO new globally-numbered R-law). Single-head scaled-dot-product attention DECOMPOSES into two
  // gem.matmuls with a softmax between them, so it is PRICED THROUGH THE EXISTING matmul roofline (NO
  // bespoke term): its (compute, mem) is the SUM of the two priced matmul costs.

  // (1) positive extents (single head: d_model == d_k; Q/K/V/out are each seq_len x d_k).
  int64_t seq = static_cast<int64_t>(getSeqLen()), dk = static_cast<int64_t>(getDK());
  if (seq < 1 || dk < 1)
    return emitOpError() << "attention: seq_len and d_k must be >= 1 (got seq_len " << seq << ", d_k "
                         << dk << ")";

  // (2) dtype known + preserved + the QUARANTINE rule: attention contains a softmax (a transcendental whose
  // libm edge returns float), so it needs an f32 result -- never i32 (mirrors gem.activation softmax).
  ::llvm::StringRef dtype = getDtype();
  if (dtype != "f32" && dtype != "i32")
    return emitOpError() << "attention: dtype '" << dtype << "' is not a known dtype (f32|i32)";
  if (dtype != "f32")
    return emitOpError() << "attention: contains a softmax (a transcendental; the libm edge returns float), "
                         << "so it needs f32, got '" << dtype << "'";

  // (3) the DERIVED two gemm dims (the decomposition into two matmuls): the scores Q@K^T gemm is
  // (seq_len, seq_len, d_k) -> the seq x seq score matrix; the context A@V gemm is (seq_len, d_k, seq_len)
  // -> the seq x d_k context. The op's declared dims must equal them (mirrors AttentionSpec.scores_dims /
  // context_dims). This is what makes attention a COMPOSITION of two structured matmuls.
  int64_t sm = static_cast<int64_t>(getScoresM()), sn = static_cast<int64_t>(getScoresN()),
          sk = static_cast<int64_t>(getScoresK());
  int64_t cm = static_cast<int64_t>(getContextM()), cn = static_cast<int64_t>(getContextN()),
          ck = static_cast<int64_t>(getContextK());
  if (sm != seq || sn != seq || sk != dk)
    return emitOpError() << "attention: declared scores Q@K^T gemm dims (" << sm << "," << sn << "," << sk
                         << ") != (seq_len, seq_len, d_k) = (" << seq << "," << seq << "," << dk << ")";
  if (cm != seq || cn != dk || ck != seq)
    return emitOpError() << "attention: declared context A@V gemm dims (" << cm << "," << cn << "," << ck
                         << ") != (seq_len, d_k, seq_len) = (" << seq << "," << dk << "," << seq << ")";

  // the realization: each underlying matmul's tile in [1, its gemm dim], with a known loop order (identical
  // to gem.matmul's tile checks, applied to each of the two gemms).
  auto checkTile = [&](const char *which, int64_t tm, int64_t tn, int64_t tk, int64_t M, int64_t N,
                       int64_t K, ::llvm::StringRef lo) -> ::mlir::LogicalResult {
    if (tm < 1 || tm > M || tn < 1 || tn > N || tk < 1 || tk > K)
      return emitOpError() << "attention: the " << which << " gemm tile must be in [1, gemm dim] (tiles "
                           << tm << "x" << tn << "x" << tk << " vs gemm dims " << M << "x" << N << "x" << K
                           << ")";
    if (lo != "ijk" && lo != "ikj" && lo != "jik")
      return emitOpError() << "attention: the " << which << " gemm loop_order must be one of ijk|ikj|jik "
                           << "(got '" << lo << "')";
    return ::mlir::success();
  };
  if (failed(checkTile("scores", static_cast<int64_t>(getScoresTileM()),
                       static_cast<int64_t>(getScoresTileN()), static_cast<int64_t>(getScoresTileK()),
                       sm, sn, sk, getScoresLoopOrder())))
    return ::mlir::failure();
  if (failed(checkTile("context", static_cast<int64_t>(getContextTileM()),
                       static_cast<int64_t>(getContextTileN()), static_cast<int64_t>(getContextTileK()),
                       cm, cn, ck, getContextLoopOrder())))
    return ::mlir::failure();

  // the per-gemm + summed roofline invariants. Each per-gemm bottleneck rides matmul.cost_of (compute/mem
  // non-negative); the attention compute/mem are the SUM of the two priced matmul costs (NO bespoke term);
  // bottleneck == max(compute, mem) (the max,+ step). Mirrors attention.cost_of: c1+c2, m1+m2.
  int64_t sc = static_cast<int64_t>(getScoresCompute()), smm = static_cast<int64_t>(getScoresMem());
  int64_t cc = static_cast<int64_t>(getContextCompute()), cmm = static_cast<int64_t>(getContextMem());
  int64_t compute = static_cast<int64_t>(getComputeCost()), mem = static_cast<int64_t>(getMemCost()),
          bn = static_cast<int64_t>(getBottleneck());
  if (sc < 0 || smm < 0 || cc < 0 || cmm < 0 || compute < 0 || mem < 0)
    return emitOpError() << "attention: all roofline cost terms must be non-negative";
  if (compute != sc + cc)
    return emitOpError() << "attention: compute_cost must equal scores_compute + context_compute = "
                         << (sc + cc) << " (the SUM of the two priced matmuls; got " << compute << ")";
  if (mem != smm + cmm)
    return emitOpError() << "attention: mem_cost must equal scores_mem + context_mem = " << (smm + cmm)
                         << " (the SUM of the two priced matmuls; got " << mem << ")";
  int64_t mx = compute > mem ? compute : mem;
  if (bn != mx)
    return emitOpError() << "attention: bottleneck must equal max(compute_cost, mem_cost) = " << mx
                         << " (got " << bn << ")";

  // the R17 accuracy axis: the two matmuls are exact and the softmax's libm kernel is trusted, so the only
  // certified error of a QUANTIZED attention is the Q8 bridge step -- acc_bound must be 0 dense
  // (quant_bits == 0) / the 1-ULP bridge bound quantized (quant_bits > 0). (mirrors attention.cost_vector.)
  int64_t accBound = static_cast<int64_t>(getAccBound());
  int64_t quantBits = static_cast<int64_t>(getQuantBits());
  int64_t wantAcc = quantBits > 0 ? 1 : 0;
  if (accBound != wantAcc)
    return emitOpError() << "attention: acc_bound must be " << wantAcc << " (the R17 Q8-bridge bound, "
                         << (quantBits > 0 ? "1 ULP for a quantized realization" : "0 for a dense one")
                         << "), got " << accBound;
  return ::mlir::success();
}

::mlir::LogicalResult GEMEmbeddingOp::verify() {
  // Rung 5 (open-weight ladder): the embedding gather's op-level shape law (mirrors the oracle's
  // EmbeddingTable validation): positive table/gather extents, a known dtype. The gather is EXACT
  // (no arithmetic), so both dtypes are legal.
  if (static_cast<int64_t>(getVocabSize()) < 1 || static_cast<int64_t>(getDim()) < 1 ||
      static_cast<int64_t>(getNIds()) < 1)
    return emitOpError() << "embedding: vocab_size, dim and n_ids must all be >= 1";
  ::llvm::StringRef dtype = getDtype();
  if (dtype != "f32" && dtype != "i32")
    return emitOpError() << "embedding: dtype '" << dtype << "' is not a known dtype (f32|i32)";
  return ::mlir::success();
}

::mlir::LogicalResult GEMRMSNormOp::verify() {
  // Rung 5: RMSNorm's op-level law (mirrors rmsnorm_reference's validation): positive extents,
  // gamma sized to the row width, and f32 -- the rms sqrt rides the trusted libm edge (the same
  // quarantine rule as the attention softmax).
  if (static_cast<int64_t>(getRows()) < 1 || static_cast<int64_t>(getDim()) < 1)
    return emitOpError() << "rmsnorm: rows and dim must be >= 1";
  if (static_cast<int64_t>(getGammaLen()) != static_cast<int64_t>(getDim()))
    return emitOpError() << "rmsnorm: gamma has " << getGammaLen() << " entries, want dim = "
                         << getDim();
  if (getDtype() != "f32")
    return emitOpError() << "rmsnorm: dtype must be f32 (the rms sqrt rides the libm edge), got '"
                         << getDtype() << "'";
  return ::mlir::success();
}

::mlir::LogicalResult GEMRopeOp::verify() {
  // Rung 5: RoPE's op-level law (mirrors rope_reference / DecoderSpec): dim must be EVEN -- the
  // rotation pairs channels (2k, 2k+1); an odd dim has no legal pairing -- base positive,
  // pos_offset non-negative, and f32 (cos/sin ride the libm edge).
  if (static_cast<int64_t>(getRows()) < 1 || static_cast<int64_t>(getDim()) < 1)
    return emitOpError() << "rope: rows and dim must be >= 1";
  if (static_cast<int64_t>(getDim()) % 2 != 0)
    return emitOpError() << "rope: dim must be EVEN (the rotation pairs channels 2k/2k+1), got "
                         << getDim();
  if (static_cast<int64_t>(getBase()) < 1)
    return emitOpError() << "rope: base must be >= 1, got " << getBase();
  if (static_cast<int64_t>(getPosOffset()) < 0)
    return emitOpError() << "rope: pos_offset must be >= 0, got " << getPosOffset();
  if (getDtype() != "f32")
    return emitOpError() << "rope: dtype must be f32 (cos/sin ride the libm edge), got '"
                         << getDtype() << "'";
  return ::mlir::success();
}

::mlir::LogicalResult GEMContentionOp::verify() {
  // The G4 cache-line/bank-conflict CONTENTION signal (mirrors bcir/kbcir/cache_predict.py::CachePredictor;
  // the op-level analytic-model well-formedness check gem.conv/attention validate inline -- NO new globally-
  // numbered R-law). It is INFORMS-ONLY: the recomputed signal rides the CONTENTION cost axis and NEVER gates
  // legality (the two-truth quarantine -- the R1-R16 verifier reads only R8/R9, never CONTENTION).
  //
  // (1) positive access shape + frozen geometry.
  int64_t stride = static_cast<int64_t>(getStride()), eb = static_cast<int64_t>(getElemBytes()),
          count = static_cast<int64_t>(getCount());
  int64_t cl = static_cast<int64_t>(getCacheline()), banks = static_cast<int64_t>(getBanks());
  if (stride < 1 || eb < 1 || count < 1)
    return emitOpError() << "contention: access shape stride/elem_bytes/count must be >= 1 (got stride "
                         << stride << ", elem_bytes " << eb << ", count " << count << ")";
  if (cl < 1 || banks < 1)
    return emitOpError() << "contention: frozen cacheline/banks must be >= 1 (got cacheline " << cl
                         << ", banks " << banks << ")";

  // (2) the frozen analytic model (the same Q8 recompute the pass cross-checks):
  //   line_waste_q8   = (min(stride, cacheline/elem_bytes) - 1) * 256   (0 == perfect cache-line utilization)
  //   bank_conflict_q8= (gcd(stride, banks) - 1)            * 256        (0 == reaches all banks, no conflict)
  //   contention_q8   = waste + conflict                                 (unit Q8 weights, both x1.0)
  //   contention_cost = (contention_q8 * count) >> 8                     (the per-element Q8 scaled by count)
  const int64_t Q8 = 256;
  int64_t epl = std::max<int64_t>(1, cl / eb);                    // useful elements per cache line (>= 1)
  int64_t reach = std::min<int64_t>(std::max<int64_t>(1, stride), epl);
  int64_t wantWaste = (reach - 1) * Q8;
  int64_t g = std::gcd(std::max<int64_t>(1, stride), banks);
  int64_t wantConflict = (g - 1) * Q8;
  int64_t wantCont = wantWaste + wantConflict;
  int64_t wantCost = (wantCont * count) >> 8;

  int64_t waste = static_cast<int64_t>(getLineWasteQ8()),
          conflict = static_cast<int64_t>(getBankConflictQ8()),
          cont = static_cast<int64_t>(getContentionQ8()), cost = static_cast<int64_t>(getContentionCost());
  if (waste != wantWaste)
    return emitOpError() << "contention: line_waste_q8 must equal (min(stride, cacheline/elem_bytes) - 1)*256 = "
                         << wantWaste << " (got " << waste << ")";
  if (conflict != wantConflict)
    return emitOpError() << "contention: bank_conflict_q8 must equal (gcd(stride, banks) - 1)*256 = "
                         << wantConflict << " (got " << conflict << ")";
  if (cont != wantCont)
    return emitOpError() << "contention: contention_q8 must equal line_waste_q8 + bank_conflict_q8 = "
                         << wantCont << " (got " << cont << ")";
  if (cost != wantCost)
    return emitOpError() << "contention: contention_cost must equal (contention_q8 * count) >> 8 = "
                         << wantCost << " (got " << cost << ")";
  return ::mlir::success();
}

::mlir::LogicalResult GEMLayoutPivotOp::verify() {
  // The G3 SoA<->AoS layout-pivot pricing (mirrors bcir/kbcir/layout.py::plan_layout + LayoutCertificate; the
  // op-level pricing well-formedness check gem.conv/attention validate inline -- NO new globally-numbered
  // R-law). It is INFORMS-ONLY: the priced layout choice/cost informs the plan's memory cost but NEVER gates
  // legality (the same two-truth quarantine; the hard "layout changes addresses, never values" invariant is
  // the oracle's, proved bit-exact by lower.layout_kernel -- the law records only the priced choice).
  //
  // (1) the frozen memory geometry the stride-penalty terms price over must be well-formed.
  int64_t fields = static_cast<int64_t>(getFields());
  int64_t cl = static_cast<int64_t>(getCacheline()), eb = static_cast<int64_t>(getElemBytes()),
          bo = static_cast<int64_t>(getBaseOverhead()), streams = static_cast<int64_t>(getStreams()),
          vw = static_cast<int64_t>(getVectorWidth()), gp = static_cast<int64_t>(getGatherPenalty());
  if (fields < 1)
    return emitOpError() << "layout_pivot: fields (record width F) must be >= 1 (got " << fields << ")";
  if (cl < 1 || eb < 1 || bo < 1 || streams < 1 || vw < 1 || gp < 1)
    return emitOpError() << "layout_pivot: frozen geometry cacheline/elem_bytes/base_overhead/streams/"
                            "vector_width/gather_penalty must be >= 1 (got "
                         << cl << "/" << eb << "/" << bo << "/" << streams << "/" << vw << "/" << gp << ")";

  // (2) recompute the SoA-vs-AoS workload cost through the SAME closed-form stride-penalty MEMORY terms
  //     realize.candidates_for prices every claim by (taking the CHEAPEST candidate's memory axis):
  //       UNIT (contiguous) min-mem    = n*streams + ceil(n/vector_width)*streams*base_overhead*1
  //       STRIDED (step F) min-mem     = n*streams + n*streams*base_overhead*min(F, cacheline/elem_bytes)
  //     A single-field sweep is UNIT under SoA / STRIDED(F) under AoS; a whole-record access is the mirror.
  //     The workload cost under a layout is the sum over the two shapes (each weighted by its records).
  int64_t sfr = static_cast<int64_t>(getSingleFieldRecords()),
          wrr = static_cast<int64_t>(getWholeRecordRecords());
  if (sfr < 0 || wrr < 0)
    return emitOpError() << "layout_pivot: record counts must be non-negative (got single_field " << sfr
                         << ", whole_record " << wrr << ")";
  int64_t sp = std::min<int64_t>(std::max<int64_t>(2, fields), std::max<int64_t>(1, cl / eb)); // STRIDED penalty
  auto unitMem = [&](int64_t n) -> int64_t {
    if (n <= 0)
      return 0;
    int64_t ceil = (n + vw - 1) / vw;
    return n * streams + ceil * streams * bo; // base_overhead * 1 (unit stride)
  };
  auto stridedMem = [&](int64_t n) -> int64_t {
    if (n <= 0)
      return 0;
    // the cheapest STRIDED candidate = min(strided penalty min(F, cl/eb), gather penalty); a target whose
    // gather is cheaper than a large stride (e.g. PTX gp=16) prices the gather instead (mirrors candidates_for).
    int64_t strided = n * streams + n * streams * bo * sp;
    int64_t gather = n * streams + n * streams * bo * gp;
    return std::min(strided, gather);
  };
  // single-field: UNIT under SoA, STRIDED under AoS; whole-record: STRIDED under SoA, UNIT under AoS.
  int64_t wantSoa, wantAos;
  if (fields <= 1) {
    // No SoA/AoS distinction (a single-field resource): both costs equal (a clean no-op).
    wantSoa = wantAos = unitMem(sfr) + unitMem(wrr);
  } else {
    wantSoa = unitMem(sfr) + stridedMem(wrr);
    wantAos = stridedMem(sfr) + unitMem(wrr);
  }
  int64_t soa = static_cast<int64_t>(getSoaCost()), aos = static_cast<int64_t>(getAosCost());
  if (soa != wantSoa)
    return emitOpError() << "layout_pivot: soa_cost must equal the priced SoA workload memory cost = "
                         << wantSoa << " (got " << soa << ")";
  if (aos != wantAos)
    return emitOpError() << "layout_pivot: aos_cost must equal the priced AoS workload memory cost = "
                         << wantAos << " (got " << aos << ")";

  // (3) the min,+ layout selection: AoS is adopted ONLY on a STRICT win; ties keep the declared default SoA.
  ::llvm::StringRef layout = getLayout();
  if (layout != "soa" && layout != "aos")
    return emitOpError() << "layout_pivot: layout must be one of soa|aos (got '" << layout << "')";
  ::llvm::StringRef wantLayout = (aos < soa) ? "aos" : "soa";
  if (layout != wantLayout)
    return emitOpError() << "layout_pivot: layout must be the min-cost choice '" << wantLayout
                         << "' (ties keep the default soa; soa_cost " << soa << " vs aos_cost " << aos
                         << "), got '" << layout << "'";

  // (4) the gain = the declared-layout (SoA) cost minus the chosen-layout cost (>= 0; pure addressing).
  int64_t chosen = (layout == "aos") ? aos : soa;
  int64_t wantGain = soa - chosen;
  int64_t gain = static_cast<int64_t>(getGain());
  if (gain != wantGain)
    return emitOpError() << "layout_pivot: gain must equal soa_cost - chosen_cost = " << wantGain
                         << " (got " << gain << ")";
  return ::mlir::success();
}

::mlir::LogicalResult GEMFusedMatmulActivationOp::verify() {
  // The G2 fused matmul+activation epilogue (mirrors bcir/kbcir/fusion.py). It carries the matmul tile
  // plan AND the activation epilogue, so it must satisfy BOTH halves' well-formedness laws plus the
  // fusion-legality boundary (softmax is non-fusible; the gain must be a strict, positive win).
  //
  // (A) the matmul tile half (identical to GEMMatmulOp::verify): positive dims, each tile in [1, dim],
  //     a known loop order.
  int64_t m = static_cast<int64_t>(getM()), n = static_cast<int64_t>(getN()),
          k = static_cast<int64_t>(getK());
  if (m <= 0 || n <= 0 || k <= 0)
    return emitOpError() << "fused matmul dims m/n/k must be positive (got " << m << "x" << n << "x"
                         << k << ")";
  int64_t tm = static_cast<int64_t>(getTileM()), tn = static_cast<int64_t>(getTileN()),
          tk = static_cast<int64_t>(getTileK());
  if (tm < 1 || tm > m || tn < 1 || tn > n || tk < 1 || tk > k)
    return emitOpError() << "each tile extent must be in [1, dim] (tiles " << tm << "x" << tn << "x"
                         << tk << " vs dims " << m << "x" << n << "x" << k << ")";
  ::llvm::StringRef lo = getLoopOrder();
  if (lo != "ijk" && lo != "ikj" && lo != "jik")
    return emitOpError() << "loop_order must be one of ijk|ikj|jik (got '" << lo << "')";

  // (B) the activation epilogue half: a known kind, and the FUSION-LEGALITY boundary -- softmax is a
  //     last-axis ROW REDUCTION (its normalizer needs the whole row), so it can NOT be applied inline as
  //     each matmul element drops out: it is NON-FUSIBLE and rejected (mirrors fusion.is_fusible_activation
  //     / FUSIBLE_ACTIVATIONS). Only the elementwise four (relu/sigmoid/tanh/gelu) fuse.
  ::llvm::StringRef kind = getKind();
  const bool isExact = (kind == "relu");
  const bool isFusibleTrans = (kind == "sigmoid" || kind == "tanh" || kind == "gelu");
  if (kind == "softmax")
    return emitOpError() << "softmax is non-fusible (a last-axis row reduction needs the whole row "
                         << "before any output element is final, so it can not be an inline epilogue)";
  if (!isExact && !isFusibleTrans)
    return emitOpError() << "unknown fusible activation kind '" << kind
                         << "' (expected relu|sigmoid|tanh|gelu)";

  // (C) the QUARANTINE dtype rule carries THROUGH the fused epilogue: a transcendental routes a libm
  //     call (expf/tanhf) through the trusted c.call.libm: edge, which returns float -- so it needs an f32
  //     result, never i32. relu (the exact, 0-ULP integer/Q-fixed-clean activation) may be i32 OR f32.
  ::llvm::StringRef dtype = getDtype();
  if (dtype != "f32" && dtype != "i32")
    return emitOpError() << kind << ": dtype '" << dtype << "' is not a known dtype (f32|i32)";
  if (isFusibleTrans && dtype != "f32")
    return emitOpError() << kind << ": transcendental activation needs f32 (the libm edge returns float), "
                         << "got '" << dtype << "'";

  // (D) the deforestation-priced fusion DECISION (the FusionCertificate invariant): the fusion was
  //     ADOPTED, so it must be a STRICT win -- gain = unfused_score - fused_score > 0 (fusion is a no-op
  //     when it does not strictly lower the score; an adopted fusion that did not win is unsound evidence).
  int64_t unfused = static_cast<int64_t>(getUnfusedScore()),
          fused = static_cast<int64_t>(getFusedScore()), gain = static_cast<int64_t>(getGain());
  if (unfused < 0 || fused < 0)
    return emitOpError() << "fusion scores must be non-negative (unfused " << unfused << ", fused "
                         << fused << ")";
  if (gain != unfused - fused)
    return emitOpError() << "gain must equal unfused_score - fused_score = " << (unfused - fused)
                         << " (got " << gain << ")";
  if (gain <= 0)
    return emitOpError() << "an adopted fusion must STRICTLY lower the score (gain " << gain
                         << " <= 0: the deforestation discount did not win)";
  return ::mlir::success();
}

::mlir::LogicalResult GEMAutodiffOp::verify() {
  // The B3 closed-set forward autodiff DAG (mirrors bcir/kbcir/autodiff.py: CLOSED_SET / _ARITY /
  // the Tape Node list -- the op-level structural well-formedness law the sibling gem ops validate
  // inline; NO new globally-numbered R-law). Slice 5 proved differentiating any expression in the
  // closed vocabulary {const,var,neg,add,sub,mul,div,dot,select} stays in the SAME set; this verifier
  // is the law that closure underwrites: the serialized forward DAG uses ONLY that vocabulary, with the
  // correct per-op arity and a strictly-earlier (acyclic/topological) operand-index discipline.
  //
  // The fixed opcode codes (mirror autodiff._ARITY's order): 0=const 1=var 2=neg 3=add 4=sub 5=mul
  // 6=div 7=dot 8=select. _ARITY: const/var leaves = 0; neg = 1; add/sub/mul/div = 2; select = 3;
  // dot is VARIADIC (an even 2k operands). Codes 0..8 are the closed set; anything else is a FOREIGN op.
  ::llvm::ArrayRef<int64_t> opcodes = getOpcodes();
  ::llvm::ArrayRef<int64_t> arities = getArities();
  ::llvm::ArrayRef<int64_t> argBase = getArgBase();
  ::llvm::ArrayRef<int64_t> args = getArgs();
  ::llvm::ArrayRef<int64_t> consts = getConsts();
  int64_t nNodes = static_cast<int64_t>(opcodes.size());
  int64_t nInputs = static_cast<int64_t>(getNInputs());
  int64_t output = static_cast<int64_t>(getOutput());

  // (4a) the parallel per-node arrays must be equal length (one entry per node).
  if (static_cast<int64_t>(arities.size()) != nNodes ||
      static_cast<int64_t>(argBase.size()) != nNodes ||
      static_cast<int64_t>(consts.size()) != nNodes)
    return emitOpError() << "autodiff: opcodes/arities/arg_base/consts must be parallel per-node arrays "
                         << "of equal length " << nNodes << " (got arities " << arities.size()
                         << ", arg_base " << argBase.size() << ", consts " << consts.size() << ")";
  if (nNodes < 1)
    return emitOpError() << "autodiff: the DAG must have at least one node (got an empty opcodes array)";
  if (nInputs < 0)
    return emitOpError() << "autodiff: n_inputs must be non-negative (got " << nInputs << ")";

  // _ARITY for the fixed-arity ops (index = opcode); dot (7) is variadic and handled separately. -1 marks
  // the variadic / out-of-range slot. Mirrors autodiff._ARITY exactly.
  static const int64_t kArity[9] = {/*const*/ 0, /*var*/ 0, /*neg*/ 1, /*add*/ 2, /*sub*/ 2,
                                    /*mul*/ 2, /*div*/ 2, /*dot*/ -1, /*select*/ 3};
  static const char *kName[9] = {"const", "var", "neg", "add", "sub",
                                 "mul", "div", "dot", "select"};

  int64_t expectedBase = 0;   // the running prefix-sum of arities: arg_base[i] must equal it (see (3a')).
  for (int64_t i = 0; i < nNodes; ++i) {
    int64_t op = opcodes[i];
    // (1) CLOSED VOCABULARY: a code outside [0, 8] is a FOREIGN opcode (the law the closure proof underwrites).
    if (op < 0 || op > 8)
      return emitOpError() << "autodiff: node " << i << " opcode " << op
                           << " is not in the closed set {const,var,neg,add,sub,mul,div,dot,select} "
                              "(codes 0..8)";
    int64_t arity = arities[i];
    if (arity < 0)
      return emitOpError() << "autodiff: node " << i << " (" << kName[op] << ") arity must be non-negative (got "
                           << arity << ")";

    // (2) ARITY: each node's operand count matches _ARITY for its op (mirrors autodiff._ARITY). dot is variadic
    // (an even 2k operands -- k u-ids then k v-ids, per Tape.dot), so only its parity/positivity is fixed.
    if (op == 7) { // dot
      if (arity < 2 || (arity % 2) != 0)
        return emitOpError() << "autodiff: node " << i << " (dot) is variadic and must have an even arity >= 2 "
                             << "(2k operands: k u-ids then k v-ids), got " << arity;
    } else if (arity != kArity[op]) {
      return emitOpError() << "autodiff: node " << i << " (" << kName[op] << ") arity must be " << kArity[op]
                           << " (mirrors autodiff._ARITY), got " << arity;
    }

    // (4b) per-node leaf/non-leaf payload consistency: a `const` carries an integer value, a `var` carries an
    // input SLOT in [0, n_inputs); every non-leaf op carries the -1 sentinel (it has no payload). (3b) a var's
    // slot must be a real input.
    int64_t payload = consts[i];
    if (op == 1) { // var
      if (payload < 0 || payload >= nInputs)
        return emitOpError() << "autodiff: node " << i << " (var) input slot " << payload
                             << " must be in [0, n_inputs " << nInputs << ")";
    } else if (op != 0) { // a non-leaf op (not const, not var) carries no payload -> the -1 sentinel.
      if (payload != -1)
        return emitOpError() << "autodiff: node " << i << " (" << kName[op]
                             << ") is a non-leaf op and must carry the -1 const sentinel, got " << payload;
    }

    // (3a) DAG / INDEX BOUNDS: this node's operands are args[arg_base[i] : arg_base[i]+arity], and EVERY operand
    // index must reference a STRICTLY EARLIER node (< i) -- guaranteeing acyclicity + topological order (no
    // forward / self / out-of-range refs). arg_base must point inside the flat args array.
    // (3a') CONTIGUITY: arg_base[i] must be the running prefix-sum of arities, so the per-node operand
    // slices contiguously PARTITION the flat args array -- no overlap (one node reading another node's
    // operand slot) and, crucially, no GAP (an unread hole that could smuggle an out-of-range / forward
    // operand index past the bounds + strictly-earlier checks below, which only scan the in-slice entries).
    // Every faithful Tape serialization emits operands node-by-node, so arg_base IS the prefix-sum; a
    // non-contiguous arg_base is malformed relative to the oracle. (Makes the sum(arities)==len check below
    // redundant-but-harmless, and ensures EVERY args entry lands in exactly one bounds-checked slice.)
    int64_t base = argBase[i];
    if (base != expectedBase)
      return emitOpError() << "autodiff: node " << i << " arg_base " << base
                           << " must equal the running prefix-sum of arities " << expectedBase
                           << " (the per-node operand slices must contiguously partition args -- no "
                              "overlap, no gap)";
    if (base + arity > static_cast<int64_t>(args.size()))
      return emitOpError() << "autodiff: node " << i << " arg_base " << base << " + arity " << arity
                           << " is out of range of the flat args array (size " << args.size() << ")";
    expectedBase += arity;
    for (int64_t a = 0; a < arity; ++a) {
      int64_t operand = args[base + a];
      if (operand < 0 || operand >= i)
        return emitOpError() << "autodiff: node " << i << " (" << kName[op] << ") operand #" << a << " = "
                             << operand << " must reference a strictly earlier node (in [0, " << i
                             << ")) -- forward/out-of-range refs break acyclicity/topological order";
    }
  }

  // (4c) the flat args array must hold exactly sum(arities) operands (no slack / no missing operands).
  int64_t totalArity = 0;
  for (int64_t a : arities)
    totalArity += a;
  if (totalArity != static_cast<int64_t>(args.size()))
    return emitOpError() << "autodiff: the flat args array must hold exactly sum(arities) = " << totalArity
                         << " operands (got " << args.size() << ")";

  // (3c) the output node index must be a valid node.
  if (output < 0 || output >= nNodes)
    return emitOpError() << "autodiff: output node index " << output << " must be in [0, node_count " << nNodes
                         << ")";
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

::mlir::LogicalResult AsmOp::verify() {
  // ASM1 structural well-formedness (mirrors bcir/frontends/cfront/lower.py::_asm_stmt + AsmInfo;
  // the op-level check the sibling law ops run inline, NOT a new globally-numbered R-law). The
  // operands are outputs-then-inputs, so their count is out + in; the results are one per output.
  int64_t nOut = static_cast<int64_t>(getOutConstraints().size());
  int64_t nIn = static_cast<int64_t>(getInConstraints().size());
  int64_t nArgs = static_cast<int64_t>(getArgs().size());
  int64_t nResults = static_cast<int64_t>(getResults().size());
  if (nArgs != nOut + nIn)
    return emitOpError() << "args count " << nArgs << " must equal out_constraints + in_constraints = "
                         << nOut << " + " << nIn << " = " << (nOut + nIn)
                         << " (operands are outputs-then-inputs)";
  if (nResults != nOut)
    return emitOpError() << "results count " << nResults << " must equal out_constraints count " << nOut
                         << " (one result per output constraint)";
  // Each output constraint must itself be an output constraint -- LLVM write-only '=' or read-write '+'.
  // Without this, a stray non-output spelling (e.g. "r") still satisfies the count checks and lowers to a
  // result-returning llvm.inline_asm whose constraint string carries no '=' output: structurally-invalid
  // inline asm that MLIR's own verifier accepts but LLVM IR translation rejects far from the source.
  ::mlir::ArrayAttr outs = getOutConstraints();
  for (int64_t i = 0; i < nOut; ++i) {
    ::llvm::StringRef c = ::mlir::cast<::mlir::StringAttr>(outs[i]).getValue();
    if (c.empty() || (c.front() != '=' && c.front() != '+'))
      return emitOpError() << "out_constraints[" << i
                           << "] must be an output constraint beginning with '=' or '+' (got '" << c << "')";
  }
  return ::mlir::success();
}

::mlir::LogicalResult PortioOp::verify() {
  // ASM2 structural well-formedness (mirrors bcir/frontends/cfront/lower.py::_portio + emit.py::
  // _portio_stmt; the op-level check the sibling law ops run inline, NOT a new globally-numbered
  // R-law). The (direction, width) pin the operand/result arity and the in/out access width.
  int64_t width = static_cast<int32_t>(getWidth());
  if (width != 8 && width != 16 && width != 32)
    return emitOpError() << "width must be one of 8/16/32 bits (b=8, w=16, l=32), got " << width;

  int64_t nArgs = static_cast<int64_t>(getArgs().size());
  int64_t nResults = static_cast<int64_t>(getResults().size());
  if (getDirection() == PortDir::In) {
    // in{b,w,l}(port) -> value: ONE operand (the port), ONE result (the read value).
    if (nArgs != 1)
      return emitOpError() << "an 'in' port read takes exactly 1 operand (the port), got " << nArgs;
    if (nResults != 1)
      return emitOpError() << "an 'in' port read produces exactly 1 result (the read value), got "
                           << nResults;
  } else {
    // out{b,w,l}(value, port): TWO operands (value, then port -- the Linux out(value,port) order),
    // ZERO results (a void write).
    if (nArgs != 2)
      return emitOpError() << "an 'out' port write takes exactly 2 operands (value, then port -- the "
                              "Linux out(value, port) order), got "
                           << nArgs;
    if (nResults != 0)
      return emitOpError() << "an 'out' port write produces no result (a void write), got " << nResults
                           << " results";
  }

  // (optional, kept simple) the read value's / written value's integer width must match the op width
  // (i8/i16/i32) and the port must be an integer -- the value rides the byte/word/long accumulator the
  // template's `%b0/%w0/%k0` modifier names, and the port the 16-bit `%w1` (dx / `Nd`).
  ::mlir::Type valueTy =
      (getDirection() == PortDir::In) ? getResults()[0].getType() : getArgs()[0].getType();
  if (auto it = ::mlir::dyn_cast<::mlir::IntegerType>(valueTy)) {
    if (static_cast<int64_t>(it.getWidth()) != width)
      return emitOpError() << "the port value type must be the i" << width
                           << " matching the op width (got " << valueTy << ")";
  } else {
    return emitOpError() << "the port value must be an integer i" << width << " (got " << valueTy << ")";
  }
  ::mlir::Type portTy =
      (getDirection() == PortDir::In) ? getArgs()[0].getType() : getArgs()[1].getType();
  // The port must be an i16: x86 in/out address the port through the 16-bit dx register (the lowering
  // hardcodes the `Nd` constraint + the `%w1` modifier that name dx), and the cfront rail types the
  // port u16. Accepting any integer width would let a semantically-wrong port (an i1/i8/i64 address
  // silently truncated to dx) pass a verifier whose contract is u16 -- the same verifier-completeness
  // gap the bcir.asm out-constraint check closes.
  auto portIntTy = ::mlir::dyn_cast<::mlir::IntegerType>(portTy);
  if (!portIntTy || portIntTy.getWidth() != 16)
    return emitOpError() << "the I/O port operand must be an i16 (the 16-bit dx / Nd port; x86 in/out "
                            "address the port via dx, got "
                         << portTy << ")";
  return ::mlir::success();
}

// Shared well-formedness for the first-class MMIO accessors (bcir.volatile_load / volatile_store): the
// device-register VALUE must be a scalar hardware-register type (integer or float -- a vector/index/struct
// device register is rejected, since the op doc promises a scalar natural width), and the device-register
// ADDRESS must be at least pointer-width (>= 32 bits) so a too-narrow address is not silently zero-extended
// into the device pointer by the inttoptr lowering. (Mirrors the operand-discipline PortioOp::verify
// applies to its port; an op-level structural check, NOT a new globally-numbered R-law.)
static ::mlir::LogicalResult verifyMmioAccess(::mlir::Operation *op,
                                              ::mlir::Type valueTy, ::mlir::Type addrTy) {
  if (!::mlir::isa<::mlir::IntegerType, ::mlir::FloatType>(valueTy))
    return op->emitOpError()
           << "the device-register value must be a scalar integer or float (got " << valueTy << ")";
  auto ait = ::mlir::dyn_cast<::mlir::IntegerType>(addrTy);
  if (!ait || ait.getWidth() < 32)
    return op->emitOpError()
           << "the device-register address must be an integer of at least pointer width (>= 32 bits; got "
           << addrTy << ")";
  return ::mlir::success();
}

::mlir::LogicalResult VolatileLoadOp::verify() {
  return verifyMmioAccess(*this, getValue().getType(), getAddr().getType());
}

::mlir::LogicalResult VolatileStoreOp::verify() {
  return verifyMmioAccess(*this, getValue().getType(), getAddr().getType());
}

// Shared address discipline for the first-class atomic accessors (bcir.atomic_rmw / atomic_cas):
// >= 32 bits, mirroring verifyMmioAccess (a too-narrow address must not be silently zero-extended
// into the object pointer by the inttoptr lowering). Op-level structural checks, NOT a new
// globally-numbered R-law -- the ordering LAW (R5) lives on the claim rail.
static ::mlir::LogicalResult verifyAtomicAddr(::mlir::Operation *op, ::mlir::Type addrTy) {
  auto ait = ::mlir::dyn_cast<::mlir::IntegerType>(addrTy);
  if (!ait || ait.getWidth() < 32)
    return op->emitOpError()
           << "the object address must be an integer of at least pointer width (>= 32 bits; got "
           << addrTy << ")";
  return ::mlir::success();
}

::mlir::LogicalResult AtomicRMWOp::verify() {
  ::llvm::StringRef k = getKind();
  if (k != "add" && k != "sub" && k != "xor" && k != "exchange")
    return emitOpError() << "unknown atomic_rmw kind '" << k
                         << "' (one of add|sub|xor|exchange, the cfront atomic families)";
  if (!::mlir::isa<::mlir::IntegerType>(getValue().getType()))
    return emitOpError() << "the RMW value must be an integer (the cfront atomic families are "
                            "integer-typed; got " << getValue().getType() << ")";
  if (getValue().getType() != getResult().getType())
    return emitOpError() << "the RMW result must have the value type (fetch/old-value semantics; got "
                         << getResult().getType() << " vs " << getValue().getType() << ")";
  return verifyAtomicAddr(*this, getAddr().getType());
}

::mlir::LogicalResult AtomicCASOp::verify() {
  if (!::mlir::isa<::mlir::IntegerType>(getExpected().getType()))
    return emitOpError() << "the CAS comparand must be an integer (got "
                         << getExpected().getType() << ")";
  if (getExpected().getType() != getDesired().getType())
    return emitOpError() << "the CAS expected/desired operands must have the same type (got "
                         << getExpected().getType() << " vs " << getDesired().getType() << ")";
  if (getExpected().getType() != getResult().getType())
    return emitOpError() << "the CAS result must have the comparand type (old-value semantics; got "
                         << getResult().getType() << " vs " << getExpected().getType() << ")";
  return verifyAtomicAddr(*this, getAddr().getType());
}

::mlir::LogicalResult AbiContractOp::verify() {
  auto sizes = getParamSizes();
  auto aligns = getParamAligns();
  if (sizes.size() != aligns.size())
    return emitOpError() << "param_sizes (" << sizes.size() << ") and param_aligns ("
                         << aligns.size() << ") must agree in length";
  for (size_t i = 0; i < sizes.size(); ++i)
    if (sizes[i] <= 0 || aligns[i] <= 0)
      return emitOpError() << "parameter " << i
                           << " has a non-positive size/align (cannot be laid out)";
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
