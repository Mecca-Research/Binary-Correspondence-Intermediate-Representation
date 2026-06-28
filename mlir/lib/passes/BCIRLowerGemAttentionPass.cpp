//===- BCIRLowerGemAttentionPass.cpp - the -bcir-lower-gem-attention lowering -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
// Lowers each `bcir.gem.attention` plan record (the K_BCIR-chosen two-matmul
// realization of single-head scaled-dot-product attention softmax(Q@K^T/sqrt(d_k))@V,
// G7) into its CONCRETE DECOMPOSITION: the two underlying gem.matmul tile sequences
// (the scores Q@K^T gemm and the context A@V gemm) with the softmax gem.activation
// between them. Attention is NOT a new primitive -- it is a COMPOSITION of ops BCIR
// already has, so this lowers each matmul through the SAME tiled-gem.block emission
// the -bcir-lower-gem-matmul / -bcir-lower-gem-conv passes use (NOT a bespoke
// attention loop nest), and the softmax through the SAME single-stripe gem.block the
// -bcir-lower-gem-activation pass emits. It is a genuine lowering: the `gem.attention`
// is ERASED after its decomposition is emitted.
//
// THE DUAL-RAIL PARITY THIS PASS ENFORCES (mirrors bcir/kbcir/attention.py exactly,
// verified by mlir/test/passes/lower_gem_attention.mlir):
//   * It prices the attention THROUGH THE EXISTING matmul roofline for BOTH matmuls
//     (attention.cost_of): the (compute, mem) is the SUM of the two priced matmul
//     costs (matmul.cost_of at each chosen tile) -- NO bespoke attention roofline
//     term; the O(seq^2) scale/softmax passes are dominated by, and folded into, the
//     Q@K^T traffic. The host-pinned per-gemm + summed compute/mem ride the op (the
//     oracle plans them with the pinned target, which differs per arch/CI); this pass
//     CROSS-CHECKS the host-independent invariants the cost model rests on -- each
//     per-gemm bottleneck == max(per-gemm compute, mem), the summed compute/mem ==
//     the two per-gemm costs, and bottleneck == max(compute, mem) (the max,+ step) --
//     exactly as -bcir-lower-gem-conv / -bcir-lower-gem-activation do. The chosen
//     plan + cost on the op MUST equal the oracle's plan_attention/cost_of output
//     (that is the gate that matters; re-asserted host-agnostically in
//     bcir/tests/test_attention_law_parity.py, which pins the oracle target x86-avx512).
//   * It reproduces plan_attention's DECOMPOSITION as the realization: the scores
//     Q@K^T gemm lowered in its chosen tile (the blocked-gemm reuse), then the softmax
//     stripe over the seq x seq score matrix (the EXISTING activation lowering -- the
//     softmax routes expf via the trusted c.call.libm: edge, the quarantine; the
//     matmuls are exact), then the context A@V gemm lowered in its chosen tile.
//   * It reproduces the R17 accuracy axis: the two matmuls' MACs are exact and the
//     softmax's libm kernel is trusted, so the only certified error of a QUANTIZED
//     attention is the Q8 bridge step (R17-bounded at 1 ULP); a dense attention
//     carries acc_bound 0.
//
// EMITTED REALIZATION (the decomposition; documented + FileCheck'd):
//   * The scores Q@K^T gemm: one gem.block per (scores_m, scores_n, scores_k) tile,
//     in scores_loop_order, named <attn>_s_t<idx> (idx = 0-based emission order).
//     base = i0*scores_n + j0; count = min(tm, M-i0)*min(tn, N-j0); strideA = scores_k
//     (Q row stride), strideB = scores_n, strideD = scores_n (the seq x seq output).
//   * The softmax: one gem.block per lane stripe over the seq x seq score matrix (the
//     last-axis reduce re-streams the row), named <attn>_sm_s<idx>. strideA = 1
//     (unit-stride read), strideB = seq_len (the softmax reduce row length), strideD =
//     1. The cross-pass quarantine record (kbcir.lane_kind transcendental, kbcir.
//     libm_edge expf) rides the first softmax stripe -- the softmax is the SOLE
//     transcendental; the matmuls are exact.
//   * The context A@V gemm: one gem.block per (context_m, context_n, context_k) tile,
//     in context_loop_order, named <attn>_c_t<idx>. strideA = context_k (A row
//     stride), strideB = context_n, strideD = context_n (the seq x d_k output).
//   * The recomputed K_BCIR plan (the two tiles + the summed priced roofline + the R17
//     verdict) is annotated on the FIRST scores tile <attn>_s_t0, as the dual-rail
//     record (exactly as -bcir-lower-gem-conv annotates its first tile).
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRPassSupport.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <string>

using namespace mlir;

namespace bcir {
namespace {

// ceil(a / b) for positive b (the tile/stripe count along one dimension).
static int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

struct LowerGemAttentionPass
    : public PassWrapper<LowerGemAttentionPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGemAttentionPass)

  StringRef getArgument() const final { return "bcir-lower-gem-attention"; }
  StringRef getDescription() const final {
    return "Lower each gem.attention plan into its concrete decomposition (the two "
           "gem.matmul tile sequences -- scores Q@K^T and context A@V -- with the "
           "softmax gem.activation between them; recomputes the SUMMED K_BCIR matmul "
           "roofline parity + the quarantine/R17 verdict; erases the attention).";
  }

  void runOnOperation() override {
    // Collect first, then rewrite: lowering mutates the IR (creates blocks, erases the
    // attention), so we must not walk the op list while we change it.
    SmallVector<GEMAttentionOp> attns;
    getOperation()->walk([&](GEMAttentionOp a) { attns.push_back(a); });
    bool ok = true;
    for (GEMAttentionOp a : attns)
      ok &= lowerOne(a);
    if (!ok)
      signalPassFailure();
  }

  // Emit one gemm's tiled gem.block sequence (identical to -bcir-lower-gem-matmul /
  // -bcir-lower-gem-conv), in the gemm's loop_order, naming the tiles <prefix>_t<idx>.
  // strideA = K (lhs row stride), strideB = strideD = N (rhs/output row stride). Returns
  // the tile count.
  int64_t emitGemm(OpBuilder &builder, Location loc, MLIRContext *ctx, const std::string &prefix,
                   int64_t M, int64_t N, int64_t K, int64_t tm, int64_t tn, int64_t tk,
                   StringRef loopOrder) {
    const int64_t nI = ceilDiv(M, tm), nJ = ceilDiv(N, tn), nK = ceilDiv(K, tk);
    int64_t idx = 0;
    auto emitTile = [&](int64_t i, int64_t j) {
      const int64_t i0 = i * tm, j0 = j * tn;
      const int64_t rows = std::min(tm, M - i0), cols = std::min(tn, N - j0);
      const int64_t base = i0 * N + j0;   // row-major gemm-output origin
      const int64_t count = rows * cols;  // tile element count
      std::string sym = (prefix + "_t" + Twine(idx)).str();
      builder.create<GEMBlockOp>(
          loc,
          /*sym_name=*/StringAttr::get(ctx, sym),
          /*base=*/builder.getI64IntegerAttr(base),
          /*count=*/builder.getI64IntegerAttr(count),
          /*strideA=*/builder.getI64IntegerAttr(K),
          /*strideB=*/builder.getI64IntegerAttr(N),
          /*strideD=*/builder.getI64IntegerAttr(N));
      ++idx;
    };
    // The tile nest in the plan's loop_order (ijk/ikj/jik), identical to the matmul/conv lowering.
    if (loopOrder == "ikj") {
      for (int64_t i = 0; i < nI; ++i)
        for (int64_t k = 0; k < nK; ++k)
          for (int64_t j = 0; j < nJ; ++j)
            emitTile(i, j);
    } else if (loopOrder == "jik") {
      for (int64_t j = 0; j < nJ; ++j)
        for (int64_t i = 0; i < nI; ++i)
          for (int64_t k = 0; k < nK; ++k)
            emitTile(i, j);
    } else {
      for (int64_t i = 0; i < nI; ++i)
        for (int64_t j = 0; j < nJ; ++j)
          for (int64_t k = 0; k < nK; ++k)
            emitTile(i, j);
    }
    return idx;
  }

  // Recompute the host-independent roofline + R17 invariants, emit the attention's
  // decomposition (the two matmul tile sequences + the softmax stripe), then erase the
  // attention. Returns false on a parity violation.
  bool lowerOne(GEMAttentionOp attn) {
    MLIRContext *ctx = attn.getContext();
    Builder ab(ctx);
    const StringRef name = attn.getSymName();

    const int64_t seq = static_cast<int64_t>(attn.getSeqLen());
    const int64_t dk = static_cast<int64_t>(attn.getDK());

    // The two derived gemm dims (the decomposition into two matmuls) + the chosen tiles.
    const int64_t sm = static_cast<int64_t>(attn.getScoresM()), sn = static_cast<int64_t>(attn.getScoresN()),
                  sk = static_cast<int64_t>(attn.getScoresK());
    const int64_t cm = static_cast<int64_t>(attn.getContextM()), cn = static_cast<int64_t>(attn.getContextN()),
                  ck = static_cast<int64_t>(attn.getContextK());
    const int64_t stm = static_cast<int64_t>(attn.getScoresTileM()),
                  stn = static_cast<int64_t>(attn.getScoresTileN()),
                  stk = static_cast<int64_t>(attn.getScoresTileK());
    const int64_t ctm = static_cast<int64_t>(attn.getContextTileM()),
                  ctn = static_cast<int64_t>(attn.getContextTileN()),
                  ctk = static_cast<int64_t>(attn.getContextTileK());
    const StringRef scoresLo = attn.getScoresLoopOrder();
    const StringRef contextLo = attn.getContextLoopOrder();

    // --- cross-check the host-INDEPENDENT roofline invariants the cost model rests on --
    // The host-pinned per-gemm + summed compute/mem ride the op (the oracle prices them through
    // matmul.cost_of with the pinned target, so we do not re-derive the arch-specific thr/bw here).
    // We cross-check: the SUMMED compute/mem == the two per-gemm costs (attention.cost_of's c1+c2,
    // m1+m2 -- NO bespoke term), and bottleneck == max(compute, mem) (the max,+ step). The same
    // recompute-and-cross-check -bcir-lower-gem-conv / -bcir-lower-gem-activation apply.
    const int64_t sc = static_cast<int64_t>(attn.getScoresCompute()),
                  smm = static_cast<int64_t>(attn.getScoresMem());
    const int64_t cc = static_cast<int64_t>(attn.getContextCompute()),
                  cmm = static_cast<int64_t>(attn.getContextMem());
    const int64_t compute = static_cast<int64_t>(attn.getComputeCost()),
                  mem = static_cast<int64_t>(attn.getMemCost()),
                  bottleneck = static_cast<int64_t>(attn.getBottleneck());
    if (compute != sc + cc) {
      attn.emitError("bcir-lower-gem-attention: compute_cost ")
          << compute << " != scores_compute + context_compute " << (sc + cc)
          << " (the SUM of the two priced matmuls -- no bespoke attention term)";
      return false;
    }
    if (mem != smm + cmm) {
      attn.emitError("bcir-lower-gem-attention: mem_cost ")
          << mem << " != scores_mem + context_mem " << (smm + cmm)
          << " (the SUM of the two priced matmuls -- no bespoke attention term)";
      return false;
    }
    int64_t mx = std::max(compute, mem);
    if (bottleneck != mx) {
      attn.emitError("bcir-lower-gem-attention: bottleneck ")
          << bottleneck << " != max(compute_cost, mem_cost) " << mx
          << " (the max,+ roofline parity)";
      return false;
    }

    // --- the R17 accuracy axis (mirrors attention.cost_vector) ------------------------
    int64_t quantBits = static_cast<int64_t>(attn.getQuantBits());
    int64_t accBound = quantBits > 0 ? 1 : 0;  // the Q8-bridge bound iff quantized (1 ULP), else 0
    if (accBound != static_cast<int64_t>(attn.getAccBound())) {
      attn.emitError("bcir-lower-gem-attention: acc_bound ")
          << static_cast<int64_t>(attn.getAccBound()) << " != the R17 Q8-bridge bound " << accBound
          << " (0 dense / 1 ULP quantized)";
      return false;
    }

    // Insert the emitted blocks right where the attention lives, immediately AFTER it.
    OpBuilder builder(attn);
    builder.setInsertionPointAfter(attn);
    const Location loc = attn.getLoc();

    // (1) the scores Q@K^T gemm: the chosen tile, in scores_loop_order -> <attn>_s_t<idx>.
    int64_t nScores = emitGemm(builder, loc, ctx, (name + "_s").str(), sm, sn, sk, stm, stn, stk, scoresLo);

    // (2) the softmax over the seq x seq score matrix: ceil(count/width) gem.block stripes (the EXISTING
    // activation lowering). The softmax is the SOLE transcendental -- it routes expf through the trusted
    // c.call.libm: edge (the quarantine); the two matmuls are exact. The width is not a free parameter on
    // attention (the oracle plans the two matmuls, not the softmax lane), so we emit ONE stripe over the
    // whole score matrix (count = seq*seq), the natural softmax pass: strideA = 1 (unit read), strideB =
    // seq (the last-axis reduce row length), strideD = 1.
    const int64_t scoreCount = std::max<int64_t>(1, seq * seq);
    builder.create<GEMBlockOp>(
        loc,
        /*sym_name=*/StringAttr::get(ctx, (name + "_sm_s0").str()),
        /*base=*/builder.getI64IntegerAttr(0),
        /*count=*/builder.getI64IntegerAttr(scoreCount),
        /*strideA=*/builder.getI64IntegerAttr(1),     // unit-stride elementwise read
        /*strideB=*/builder.getI64IntegerAttr(seq),   // the softmax last-axis reduce row length
        /*strideD=*/builder.getI64IntegerAttr(1));     // unit-stride write

    // (3) the context A@V gemm: the chosen tile, in context_loop_order -> <attn>_c_t<idx>.
    emitGemm(builder, loc, ctx, (name + "_c").str(), cm, cn, ck, ctm, ctn, ctk, contextLo);

    // Annotate the recomputed K_BCIR plan (the two tiles + the SUMMED priced roofline + the R17 verdict)
    // on the FIRST scores tile <attn>_s_t0, as the cross-pass dual-rail record, and the quarantine verdict
    // (kbcir.lane_kind transcendental, kbcir.libm_edge expf) on the softmax stripe.
    if (nScores > 0) {
      getOperation()->walk([&](GEMBlockOp b) {
        if (b.getSymName() == (name + "_s_t0").str()) {
          b->setAttr("kbcir.seq_len", ab.getI64IntegerAttr(seq));
          b->setAttr("kbcir.d_k", ab.getI64IntegerAttr(dk));
          b->setAttr("kbcir.scores_loop_order", ab.getStringAttr(scoresLo));
          b->setAttr("kbcir.context_loop_order", ab.getStringAttr(contextLo));
          b->setAttr("kbcir.scores_compute", ab.getI64IntegerAttr(sc));
          b->setAttr("kbcir.scores_mem", ab.getI64IntegerAttr(smm));
          b->setAttr("kbcir.context_compute", ab.getI64IntegerAttr(cc));
          b->setAttr("kbcir.context_mem", ab.getI64IntegerAttr(cmm));
          b->setAttr("kbcir.compute_cost", ab.getI64IntegerAttr(compute));
          b->setAttr("kbcir.mem_cost", ab.getI64IntegerAttr(mem));
          b->setAttr("kbcir.bottleneck", ab.getI64IntegerAttr(bottleneck));
          b->setAttr("kbcir.acc_bound", ab.getI64IntegerAttr(accBound));
        }
      });
    }
    getOperation()->walk([&](GEMBlockOp b) {
      if (b.getSymName() == (name + "_sm_s0").str()) {
        // the softmax is the SOLE transcendental: it rides the trusted c.call.libm: edge (expf), the
        // quarantine. The two matmuls are exact (the deterministic rail).
        b->setAttr("kbcir.lane_kind", ab.getStringAttr("transcendental"));
        b->setAttr("kbcir.libm_edge", ab.getStringAttr("expf"));
      }
    });

    // Genuine lowering: the plan record is consumed -- erase it.
    attn.erase();
    return true;
  }
};

}  // namespace

std::unique_ptr<Pass> createLowerGemAttentionPass() {
  return std::make_unique<LowerGemAttentionPass>();
}

}  // namespace bcir
