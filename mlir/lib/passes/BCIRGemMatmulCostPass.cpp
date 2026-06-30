//===- BCIRGemMatmulCostPass.cpp - the -bcir-gem-matmul-cost roofline ------*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
// Recomputes the B1 matmul ROOFLINE COST for each `bcir.gem.matmul` plan op (the
// law-rail port of bcir/kbcir/matmul.py::cost_of) and annotates the recomputed
// compute/mem/bottleneck terms on the op. Like -bcir-layout-pivot / -bcir-cache-
// contention this is a cost PRODUCER, not a lowering (cf. -bcir-lower-gem-matmul,
// which ERASES the matmul into its tiles): the op is NOT erased -- the recomputed
// roofline is annotated as kbcir.* attrs (the cross-pass dual-rail record).
//
// THE ROOFLINE THIS PASS REPRODUCES (mirrors matmul.py::cost_of exactly, verified by
// mlir/test/passes/gem_matmul_cost.mlir + bcir/tests/test_matmul_law_parity.py):
//   A matmul is M*N*K MACs streaming the A/B/C tiles. The dual-semiring search prices
//   each tiling through two analytic roofline terms and binds the max,+ bottleneck:
//     compute = ceilDiv(M*N*K, vector_width*(2 if fma else 1))   -- the FLOP-throughput term
//     a_reads = M*K*ceil(N/tile_n)   (A re-read once per N-tile -- the blocked-GEMM reuse)
//     b_reads = K*N*ceil(M/tile_m)   (B re-read once per M-tile)
//     c_traffic = M*N*(ceil(K/tile_k) + 1)
//     mem     = ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels) -- the bandwidth term
//     bottleneck = max(compute, mem)                              -- the max,+ binding resource
//   working_set = tile_m*tile_k + tile_k*tile_n + tile_m*tile_n must fit the modeled per-core
//   cache budget = (512 * cacheline) / elem_bytes elements (fits_cache).
//
// DETERMINISM (host-independence): the oracle cost depends on the TargetProfile, so this pass
// PINS the reference constants to TargetProfile.x86_avx512() (named consts below), EXACTLY as
// the sibling cost passes + their *_law_parity.py tests pin x86-avx512 so the recomputed cost
// is CI-host-independent (TargetProfile.for_host() would price the roofline differently per arch).
//
// THE PARITY THE VERIFIER OMITS: GEMMatmulOp::verify() checks only well-formedness (positive
// dims, each tile in [1, dim], a known loop order, bottleneck == max(compute, mem)) -- it does
// NOT recompute the ANALYTIC cost from the shape/tile. This pass adds exactly that: it ports
// cost_of, cross-checks the declared compute_cost/mem_cost/bottleneck, and emitError on a
// mismatch (the dual-rail R13 parity gate). INFORMS-ONLY: the recomputed roofline informs the
// plan's cost but is NEVER read by -bcir-verify as a legality verdict -- the matmul op is NOT
// erased and never gates legality (the same two-truth quarantine the sibling cost passes keep).
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

#include <algorithm>

using namespace mlir;

namespace bcir {
namespace {

// ceil(a / b) for positive b (mirrors matmul.py's (x + d - 1)//d / math.ceil and the
// BCIRLowerGemMatmulPass tile-count helper).
static int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

struct GemMatmulCostPass
    : public PassWrapper<GemMatmulCostPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemMatmulCostPass)

  StringRef getArgument() const final { return "bcir-gem-matmul-cost"; }
  StringRef getDescription() const final {
    return "Recompute the B1 matmul roofline (port of matmul.py::cost_of) for each gem.matmul "
           "op and parity-check the declared compute/mem/bottleneck; annotate kbcir.* "
           "INFORMS-ONLY (never a legality verdict).";
  }

  void runOnOperation() override {
    SmallVector<GEMMatmulOp> ops;
    getOperation()->walk([&](GEMMatmulOp mm) { ops.push_back(mm); });
    bool ok = true;
    for (GEMMatmulOp mm : ops)
      ok &= priceOne(mm);
    if (!ok)
      signalPassFailure();
  }

  // Recompute the roofline cost from the op's shape + tiling through the pinned x86-avx512
  // reference constants, cross-check the declared compute/mem/bottleneck, and annotate the
  // recomputed roofline. Returns false on a parity violation. (The op is NOT erased -- this is
  // a cost producer, not a lowering.)
  bool priceOne(GEMMatmulOp op) {
    MLIRContext *ctx = op.getContext();
    Builder ab(ctx);

    // The pinned TargetProfile.x86_avx512() reference constants (matmul.py::cost_of reads these
    // off the target; pinned here for CI-host-independence, EXACTLY as the sibling cost passes +
    // their *_law_parity.py tests pin x86-avx512): vector_width = max(1,8,16) = 16, fma = true,
    // mem_unit = 1, mem_channels = 4, cacheline = 64, elem_bytes = 4.
    const int64_t kVectorWidth = 16;
    const int64_t kFma = 2;           // (2 if fma else 1); avx512 has fma -> 2
    const int64_t kMemUnit = 1;
    const int64_t kMemChannels = 4;
    const int64_t kCacheline = 64;
    const int64_t kElemBytes = 4;

    int64_t M = static_cast<int64_t>(op.getM());
    int64_t N = static_cast<int64_t>(op.getN());
    int64_t K = static_cast<int64_t>(op.getK());
    int64_t tm = std::max<int64_t>(1, static_cast<int64_t>(op.getTileM()));
    int64_t tn = std::max<int64_t>(1, static_cast<int64_t>(op.getTileN()));
    int64_t tk = std::max<int64_t>(1, static_cast<int64_t>(op.getTileK()));

    // --- recompute the analytic roofline (identical to matmul.py::cost_of) ---------------
    // Computed in int64 -- the cost domain the GEMMatmulOp attrs themselves are declared in (I64Attr) and
    // that the verifier + every sibling cost pass share. A matmul whose M*N*K (or ~3*M*N*K traffic) would
    // exceed int64 -- a square dim past ~1.45M, i.e. multi-terabyte operands no real GEMM approaches -- is
    // outside this representable domain anyway (its own declared cost could not fit i64); such a shape would
    // trip the parity gate (a loud spurious error), never annotate a silently-wrapped cost.
    int64_t macs = M * N * K;
    int64_t thr = std::max<int64_t>(1, kVectorWidth * kFma);     // the FLOP-throughput divisor
    int64_t compute = ceilDiv(macs, thr);                        // M*N*K MACs / (vector_width * fma)
    int64_t aReads = M * K * ceilDiv(N, tn);                     // A reused across the columns in a tile
    int64_t bReads = K * N * ceilDiv(M, tm);                     // B reused across the rows in a tile
    int64_t cTraffic = M * N * (ceilDiv(K, tk) + 1);             // C streamed once per K-tile, plus the write
    int64_t bw = std::max<int64_t>(1, kMemUnit * kMemChannels);  // the bandwidth divisor
    int64_t mem = ceilDiv(aReads + bReads + cTraffic, bw);       // bytes streamed / bandwidth
    int64_t bottleneck = std::max(compute, mem);                 // max,+ : the binding roofline resource
    int64_t cacheBudget =
        (512 * kCacheline) / std::max<int64_t>(1, kElemBytes);   // the modeled per-core working-set budget
    int64_t workingSet = tm * tk + tk * tn + tm * tn;            // a tile of A, B, C must co-reside
    bool fitsCache = workingSet <= cacheBudget;

    // --- cross-check the declared roofline (the dual-rail parity gate) ------------------
    if (compute != static_cast<int64_t>(op.getComputeCost())) {
      op.emitError("bcir-gem-matmul-cost: compute_cost ")
          << static_cast<int64_t>(op.getComputeCost()) << " != the recomputed roofline compute "
          << compute << " (ceilDiv(M*N*K, vector_width*fma))";
      return false;
    }
    if (mem != static_cast<int64_t>(op.getMemCost())) {
      op.emitError("bcir-gem-matmul-cost: mem_cost ")
          << static_cast<int64_t>(op.getMemCost()) << " != the recomputed roofline mem " << mem
          << " (ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels))";
      return false;
    }
    if (bottleneck != static_cast<int64_t>(op.getBottleneck())) {
      op.emitError("bcir-gem-matmul-cost: bottleneck ")
          << static_cast<int64_t>(op.getBottleneck()) << " != max(compute, mem) " << bottleneck;
      return false;
    }

    // --- annotate the recomputed roofline (the cross-pass dual-rail record) -------------
    // INFORMS-ONLY: the recomputed roofline informs the plan's cost but never gates legality
    // (-bcir-verify never reads these). fits_cache mirrors the TilePlan's modeled budget.
    op->setAttr("kbcir.compute_cost", ab.getI64IntegerAttr(compute));
    op->setAttr("kbcir.mem_cost", ab.getI64IntegerAttr(mem));
    op->setAttr("kbcir.bottleneck", ab.getI64IntegerAttr(bottleneck));
    op->setAttr("kbcir.fits_cache", ab.getBoolAttr(fitsCache));
    op->setAttr("kbcir.informs_only", ab.getBoolAttr(true));         // never a legality verdict
    return true;
  }
};

}  // namespace

std::unique_ptr<Pass> createGemMatmulCostPass() {
  return std::make_unique<GemMatmulCostPass>();
}

}  // namespace bcir
