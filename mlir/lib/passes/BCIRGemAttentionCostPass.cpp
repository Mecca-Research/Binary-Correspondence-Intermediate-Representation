//===- BCIRGemAttentionCostPass.cpp - the -bcir-gem-attention-cost roofline ---*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
// Recomputes the G7 attention ROOFLINE COST for each `bcir.gem.attention` plan op (the
// law-rail port of bcir/kbcir/attention.py::cost_of) and annotates the recomputed
// per-gemm + summed compute/mem/bottleneck terms on the op. Like -bcir-gem-matmul-cost /
// -bcir-gem-activation-cost / -bcir-gem-conv-cost / -bcir-layout-pivot / -bcir-cache-
// contention this is a cost PRODUCER, not a lowering (cf. -bcir-lower-gem-attention, which
// ERASES the attention into its two gem.matmul tile sequences + the softmax stripe): the op
// is NOT erased -- the recomputed roofline is annotated as kbcir.* attrs (the cross-pass
// dual-rail record).
//
// THE ROOFLINE THIS PASS REPRODUCES (mirrors attention.py::cost_of exactly, verified by
// mlir/test/passes/gem_attention_cost.mlir + bcir/tests/test_attention_law_parity.py):
//   Single-head scaled-dot-product attention softmax(Q@K^T/sqrt(d_k))@V is NOT a new primitive:
//   it DECOMPOSES into the two gem.matmuls BCIR already has (a softmax + an exact scale ride
//   between them) -- the scores Q@K^T gemm = (seq, seq, d_k) and the context A@V gemm =
//   (seq, d_k, seq). So attention is PRICED THROUGH THE EXISTING matmul roofline for BOTH
//   matmuls (NO bespoke attention term): attention.cost_of calls matmul.cost_of on each gemm
//   at its chosen tile and SUMS them (the O(seq^2) scale/softmax passes are dominated by, and
//   folded into, the Q@K^T traffic -- attention.cost_of adds NO softmax term, exactly as the
//   oracle: it returns c1+c2, m1+m2, f1 and f2). For each gemm, matmul.cost_of is:
//     compute   = ceilDiv(M*N*K, vector_width*(2 if fma else 1))   -- the FLOP-throughput term
//     a_reads   = M*K*ceil(N/tile_n)   (A re-read once per N-tile -- the blocked-GEMM reuse)
//     b_reads   = K*N*ceil(M/tile_m)   (B re-read once per M-tile)
//     c_traffic = M*N*(ceil(K/tile_k) + 1)
//     mem       = ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels) -- the bandwidth term
//   and the attention totals are:
//     compute_cost = scores_compute + context_compute   (the SUM of the two priced matmuls)
//     mem_cost     = scores_mem     + context_mem        (the SUM of the two priced matmuls)
//     bottleneck   = max(compute_cost, mem_cost)         -- the max,+ binding resource
//   fits_cache is the AND of the two gemms' working-set fits (both gemms must co-reside).
//
// DETERMINISM (host-independence): the oracle cost depends on the TargetProfile, so this pass
// PINS the reference constants to TargetProfile.x86_avx512() (named consts below), EXACTLY as
// the sibling cost passes + their *_law_parity.py tests pin x86-avx512 so the recomputed cost
// is CI-host-independent (TargetProfile.for_host() would price the roofline differently per arch).
//
// THE PARITY THE VERIFIER OMITS: GEMAttentionOp::verify() checks only well-formedness (positive
// seq_len/d_k, a known + preserved dtype that is f32 -- the softmax quarantine rule, the DERIVED
// two gemm dims internally consistent with (seq_len, d_k), each gemm tile in [1, gemm dim] with a
// known loop order, all cost terms non-negative, the SUMMED compute/mem == the two declared
// per-gemm costs, bottleneck == max(compute, mem), and the R17 acc_bound) -- it does NOT recompute
// the ANALYTIC per-gemm cost from the gemm dims/tile. This pass adds exactly that: it ports cost_of
// (matmul.cost_of on each gemm, summed), cross-checks the declared per-gemm scores/context
// compute/mem AND the summed compute_cost/mem_cost/bottleneck, and emitError on a mismatch (the
// dual-rail R13 parity gate). INFORMS-ONLY: the recomputed roofline informs the plan's cost but is
// NEVER read by -bcir-verify as a legality verdict -- the attention op is NOT erased and never gates
// legality (the same two-truth quarantine the sibling cost passes keep).
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIR/BCIRPasses.h"
#include "BCIRPassSupport.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>

using namespace mlir;

namespace bcir {
namespace {

// ceil(a / b) for positive b (mirrors attention.py / matmul.py's (x + d - 1)//d / math.ceil and
// the sibling cost-pass ceilDiv helpers).
static int64_t ceilDiv(int64_t a, int64_t b) {
  return a / b + (a % b != 0);
}

struct GemAttentionCostPass : public PassWrapper<GemAttentionCostPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemAttentionCostPass)

  StringRef getArgument() const final { return "bcir-gem-attention-cost"; }
  StringRef getDescription() const final {
    return "Recompute the G7 attention roofline (port of attention.py::cost_of -- attention is a "
           "COMPOSITION of two gem.matmuls priced through matmul.cost_of and SUMMED, no bespoke "
           "term) for each gem.attention op and parity-check the declared per-gemm + summed "
           "compute/mem/bottleneck; annotate kbcir.* INFORMS-ONLY (never a legality verdict).";
  }

  void runOnOperation() override {
    SmallVector<GEMAttentionOp> ops;
    getOperation()->walk([&](GEMAttentionOp at) { ops.push_back(at); });
    bool ok = true;
    for (GEMAttentionOp at : ops)
      ok &= priceOne(at);
    if (!ok)
      signalPassFailure();
  }

  // The pinned-x86-avx512 matmul roofline for one gemm (M,N,K) at tile (tm,tn,tk) -- IDENTICAL to
  // matmul.py::cost_of (the helper -bcir-gem-matmul-cost / -bcir-gem-conv-cost recompute). Writes
  // the (compute, mem) terms through the out params + returns the working-set fits_cache flag.
  static bool gemmCostOf(int64_t M, int64_t N, int64_t K, int64_t tm, int64_t tn, int64_t tk,
                         int64_t &compute, int64_t &mem, bool &fitsCache) {
    // The pinned TargetProfile.x86_avx512() reference constants (attention.py::cost_of -> matmul.
    // cost_of reads these off the target; pinned here for CI-host-independence, EXACTLY as the
    // sibling cost passes + their *_law_parity.py tests pin x86-avx512): vector_width = max(1,8,16)
    // = 16, fma = true, mem_unit = 1, mem_channels = 4, cacheline = 64, elem_bytes = 4.
    const int64_t kVectorWidth = 16;
    const int64_t kFma = 2; // (2 if fma else 1); avx512 has fma -> 2
    const int64_t kMemUnit = 1;
    const int64_t kMemChannels = 4;
    const int64_t kCacheline = 64;
    const int64_t kElemBytes = 4;

    int64_t macs, traffic, workingSet;
    if (!checkedGemmDerived(M, N, K, tm, tn, tk, macs, traffic, workingSet))
      return false;
    int64_t thr = std::max<int64_t>(1, kVectorWidth * kFma);    // the FLOP-throughput divisor
    compute = ceilDiv(macs, thr);                               // M*N*K MACs / (vector_width * fma)
    int64_t bw = std::max<int64_t>(1, kMemUnit * kMemChannels); // the bandwidth divisor
    mem = ceilDiv(traffic, bw);                                 // bytes streamed / bandwidth
    int64_t cacheBudget =
        (512 * kCacheline) /
        std::max<int64_t>(1, kElemBytes); // the modeled per-core working-set budget
    fitsCache = workingSet <= cacheBudget;
    return true;
  }

  // Recompute the attention roofline cost from the op's two equivalent gemm dims + tilings through
  // the pinned x86-avx512 reference constants (each gemm priced through matmul.cost_of -- identical
  // to attention.cost_of's two matmul.cost_of calls -- then SUMMED), cross-check the declared
  // per-gemm + summed compute/mem/bottleneck, and annotate the recomputed roofline. Returns false on
  // a parity violation. (The op is NOT erased -- this is a cost producer, not a lowering.)
  bool priceOne(GEMAttentionOp op) {
    MLIRContext *ctx = op.getContext();
    Builder ab(ctx);

    // the DERIVED two gemm dims (carried for the verifier to cross-check the decomposition):
    //   scores Q@K^T : (seq_len, seq_len, d_k);  context A@V : (seq_len, d_k, seq_len).
    int64_t sm = static_cast<int64_t>(op.getScoresM()), sn = static_cast<int64_t>(op.getScoresN()),
            sk = static_cast<int64_t>(op.getScoresK());
    int64_t cm = static_cast<int64_t>(op.getContextM()),
            cn = static_cast<int64_t>(op.getContextN()),
            ck = static_cast<int64_t>(op.getContextK());
    int64_t stm = std::max<int64_t>(1, static_cast<int64_t>(op.getScoresTileM()));
    int64_t stn = std::max<int64_t>(1, static_cast<int64_t>(op.getScoresTileN()));
    int64_t stk = std::max<int64_t>(1, static_cast<int64_t>(op.getScoresTileK()));
    int64_t ctm = std::max<int64_t>(1, static_cast<int64_t>(op.getContextTileM()));
    int64_t ctn = std::max<int64_t>(1, static_cast<int64_t>(op.getContextTileN()));
    int64_t ctk = std::max<int64_t>(1, static_cast<int64_t>(op.getContextTileK()));

    // --- recompute the analytic roofline (identical to attention.py::cost_of == two matmul.cost_of
    // calls, SUMMED) -------------------------------------------------------------------------------
    // Computed in int64 -- the cost domain the GEMAttentionOp attrs themselves are declared in (I64Attr)
    // and that the verifier + every sibling cost pass share. An attention whose per-gemm M*N*K (or
    // ~3*M*N*K traffic) would exceed int64 -- a gemm dim past ~1.45M, i.e. multi-terabyte operands no
    // real attention approaches -- is outside this representable domain anyway (its own declared cost
    // could not fit i64); such a shape would trip the parity gate (a loud spurious error), never
    // annotate a silently-wrapped cost.
    int64_t scoresCompute = 0, scoresMem = 0, contextCompute = 0, contextMem = 0;
    bool scoresFits = false, contextFits = false;
    if (!gemmCostOf(sm, sn, sk, stm, stn, stk, scoresCompute, scoresMem, scoresFits) ||
        !gemmCostOf(cm, cn, ck, ctm, ctn, ctk, contextCompute, contextMem, contextFits)) {
      op.emitError(
          "bcir-gem-attention-cost: derived per-gemm roofline cost exceeds signed 64-bit range");
      return false;
    }
    int64_t compute, mem;
    if (!checkedAddNonnegative(scoresCompute, contextCompute, compute) ||
        !checkedAddNonnegative(scoresMem, contextMem, mem)) {
      op.emitError("bcir-gem-attention-cost: summed roofline cost exceeds signed 64-bit range");
      return false;
    }
    int64_t bottleneck = std::max(compute, mem); // max,+ : the binding roofline resource
    bool fitsCache = scoresFits && contextFits;  // both gemms must fit (f1 and f2)

    // --- cross-check the declared roofline (the dual-rail parity gate) ------------------
    // the PER-GEMM priced matmul costs (each == matmul.cost_of at its chosen tile).
    if (scoresCompute != static_cast<int64_t>(op.getScoresCompute())) {
      op.emitError("bcir-gem-attention-cost: scores_compute ")
          << static_cast<int64_t>(op.getScoresCompute())
          << " != the recomputed scores Q@K^T roofline compute " << scoresCompute
          << " (ceilDiv(scores_m*scores_n*scores_k, vector_width*fma))";
      return false;
    }
    if (scoresMem != static_cast<int64_t>(op.getScoresMem())) {
      op.emitError("bcir-gem-attention-cost: scores_mem ")
          << static_cast<int64_t>(op.getScoresMem())
          << " != the recomputed scores Q@K^T roofline mem " << scoresMem
          << " (ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels))";
      return false;
    }
    if (contextCompute != static_cast<int64_t>(op.getContextCompute())) {
      op.emitError("bcir-gem-attention-cost: context_compute ")
          << static_cast<int64_t>(op.getContextCompute())
          << " != the recomputed context A@V roofline compute " << contextCompute
          << " (ceilDiv(context_m*context_n*context_k, vector_width*fma))";
      return false;
    }
    if (contextMem != static_cast<int64_t>(op.getContextMem())) {
      op.emitError("bcir-gem-attention-cost: context_mem ")
          << static_cast<int64_t>(op.getContextMem())
          << " != the recomputed context A@V roofline mem " << contextMem
          << " (ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels))";
      return false;
    }
    // the SUMMED roofline (compute = the two computes summed, mem = the two mems summed -- attention
    // is a COMPOSITION of two gem.matmuls; NO bespoke term).
    if (compute != static_cast<int64_t>(op.getComputeCost())) {
      op.emitError("bcir-gem-attention-cost: compute_cost ")
          << static_cast<int64_t>(op.getComputeCost()) << " != the recomputed roofline compute "
          << compute << " (scores_compute + context_compute -- the SUM of the two priced matmuls)";
      return false;
    }
    if (mem != static_cast<int64_t>(op.getMemCost())) {
      op.emitError("bcir-gem-attention-cost: mem_cost ")
          << static_cast<int64_t>(op.getMemCost()) << " != the recomputed roofline mem " << mem
          << " (scores_mem + context_mem -- the SUM of the two priced matmuls)";
      return false;
    }
    if (bottleneck != static_cast<int64_t>(op.getBottleneck())) {
      op.emitError("bcir-gem-attention-cost: bottleneck ")
          << static_cast<int64_t>(op.getBottleneck()) << " != max(compute, mem) " << bottleneck;
      return false;
    }

    // --- annotate the recomputed roofline (the cross-pass dual-rail record) -------------
    // INFORMS-ONLY: the recomputed roofline informs the plan's cost but never gates legality
    // (-bcir-verify never reads these). fits_cache mirrors the AND of the two gemms' modeled budgets.
    op->setAttr("kbcir.scores_compute", ab.getI64IntegerAttr(scoresCompute));
    op->setAttr("kbcir.scores_mem", ab.getI64IntegerAttr(scoresMem));
    op->setAttr("kbcir.context_compute", ab.getI64IntegerAttr(contextCompute));
    op->setAttr("kbcir.context_mem", ab.getI64IntegerAttr(contextMem));
    op->setAttr("kbcir.compute_cost", ab.getI64IntegerAttr(compute));
    op->setAttr("kbcir.mem_cost", ab.getI64IntegerAttr(mem));
    op->setAttr("kbcir.bottleneck", ab.getI64IntegerAttr(bottleneck));
    op->setAttr("kbcir.fits_cache", ab.getBoolAttr(fitsCache));
    op->setAttr("kbcir.informs_only", ab.getBoolAttr(true)); // never a legality verdict
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> createGemAttentionCostPass() {
  return std::make_unique<GemAttentionCostPass>();
}

} // namespace bcir
