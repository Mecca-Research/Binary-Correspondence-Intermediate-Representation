//===- BCIRSensePass.cpp - regret-driven telemetry resolution gate -*- C++ -*-===//
//
// -bcir-sense: the deterministic telemetry-sensing gate (kbcir/sensing.py RegretSensor.sense).
// It aggregates the bcir.trace.data_dna records per segment into streaming integer sufficient
// statistics (n, Sigma cycles, Sigma cycles^2), computes each segment's coefficient of
// variation cv_milli = 1000*stdev/mean (population variance, floor-isqrt stdev -- the exact
// integer formula of sensing.PathSamples), then assigns a telemetry resolution: the most
// uncertain segments (n >= min_samples AND cv_milli >= threshold) get "high" up to a budget,
// the rest "low" (enough samples to trust the estimate) or "off" (too few). Deterministic:
// ranked by (-cv_milli, segment). Annotates each record with kbcir.sense_resolution +
// kbcir.sense_cv. This is the a-posteriori telemetry instrument; the a-priori ranker-margin
// variant (sense_by_ranker) stays off-rail (it leans on a learned confidence).
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRPassSupport.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace mlir;

namespace bcir {
namespace {

// sensing.py defaults: the CV (relative-uncertainty) threshold in milli, the high-res budget
// (overhead cap), and the minimum samples to trust an estimate.
static constexpr int64_t kCvThresholdMilli = 200;
static constexpr int64_t kHiresBudget = 4;
static constexpr int64_t kMinSamples = 2;

// floor(sqrt(n)) for n >= 0 (math.isqrt): a double estimate, integer-corrected both ways.
static int64_t isqrt64(int64_t n) {
  if (n <= 0)
    return 0;
  int64_t x = static_cast<int64_t>(std::sqrt(static_cast<double>(n)));
  while (x > 0 && x > n / x)
    --x;
  while (x + 1 <= n / (x + 1))
    ++x;
  return x;
}

struct SensePass : public PassWrapper<SensePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SensePass)

  StringRef getArgument() const final { return "bcir-sense"; }
  StringRef getDescription() const final {
    return "Regret-driven telemetry-sensing gate (kbcir/sensing.py): per-segment cv_milli over "
           "the trace.data_dna records -> high/low/off resolution; annotates "
           "kbcir.sense_resolution / kbcir.sense_cv.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b);
    });
  }

  void runOnModule(Operation *root, Builder &b) {
    struct Stats {
      int64_t n = 0, total = 0, totalSq = 0;
      bool overflow = false;
    };
    llvm::MapVector<StringRef, Stats> bySeg;   // segment -> sufficient stats (first-seen order)
    SmallVector<TraceDataDnaOp> records;
    root->walk([&](TraceDataDnaOp d) {
      int64_t c = static_cast<int64_t>(d.getCycles());
      Stats &s = bySeg[d.getSegment()];
      int64_t nextN, nextTotal, square, nextTotalSq;
      const bool sumsWereValid = !s.overflow;
      // Keep the sample count accurate even after a sum/square becomes
      // unrepresentable.  The count decides whether the conservative
      // max-uncertainty result is eligible for high-resolution telemetry.
      if (!checkedAddNonnegative(s.n, 1, nextN)) {
        s.overflow = true;
        s.n = std::numeric_limits<int64_t>::max();
      } else {
        s.n = nextN;
      }
      if (sumsWereValid && !s.overflow) {
        if (c < 0 || !checkedAddNonnegative(s.total, c, nextTotal) ||
            !checkedMulNonnegative(c, c, square) ||
            !checkedAddNonnegative(s.totalSq, square, nextTotalSq)) {
          s.overflow = true;
        } else {
          s.total = nextTotal;
          s.totalSq = nextTotalSq;
        }
      }
      records.push_back(d);
    });
    if (records.empty())
      return;

    struct SegInfo {
      StringRef seg;
      int64_t n, cv;
    };
    SmallVector<SegInfo> segs;
    for (auto &kv : bySeg) {
      const Stats &s = kv.second;
      if (s.overflow) {
        segs.push_back({kv.first, s.n, std::numeric_limits<int64_t>::max()});
        continue;
      }
      int64_t mean = s.n ? s.total / s.n : 0;
      long double numerator = static_cast<long double>(s.n) * s.totalSq -
                              static_cast<long double>(s.total) * s.total;
      long double denominator = static_cast<long double>(s.n) * s.n;
      long double variance = denominator > 0 ? numerator / denominator : 0;
      variance = std::max<long double>(0, variance);
      int64_t var = variance >= std::numeric_limits<int64_t>::max()
                        ? std::numeric_limits<int64_t>::max()
                        : static_cast<int64_t>(variance);
      int64_t stdev = isqrt64(var);
      int64_t cv = mean ? saturatingMulNonnegative(1000, stdev) / mean : 0;
      segs.push_back({kv.first, s.n, cv});
    }
    // Rank: most uncertain first (-cv_milli), then segment name (the deterministic tie-break).
    llvm::sort(segs, [](const SegInfo &a, const SegInfo &b) {
      return a.cv != b.cv ? a.cv > b.cv : a.seg < b.seg;
    });

    llvm::StringMap<StringRef> resOf;
    llvm::StringMap<int64_t> cvOf;
    int64_t hires = 0;
    for (const SegInfo &s : segs) {
      StringRef res;
      bool uncertain = s.n >= kMinSamples && s.cv >= kCvThresholdMilli;
      if (uncertain && hires < kHiresBudget) {
        res = "high";
        ++hires;
      } else if (s.n >= kMinSamples) {
        res = "low";
      } else {
        res = "off";
      }
      resOf[s.seg] = res;
      cvOf[s.seg] = s.cv;
    }

    for (TraceDataDnaOp d : records) {
      StringRef seg = d.getSegment();
      d->setAttr("kbcir.sense_resolution", b.getStringAttr(resOf[seg]));
      d->setAttr("kbcir.sense_cv", b.getI64IntegerAttr(cvOf[seg]));
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createSensePass() {
  return std::make_unique<SensePass>();
}

}  // namespace bcir
