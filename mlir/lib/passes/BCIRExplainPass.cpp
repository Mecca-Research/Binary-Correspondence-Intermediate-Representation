//===- BCIRExplainPass.cpp - proof-carrying decision record as IR annotations -*- C++ -*-===//
//
// -bcir-explain: the law-rail port of bcir/kbcir/proof.explain. Where -bcir-plan annotates
// only the chosen plan, this pass attaches the *rationale* -- the proof-carrying decision
// record -- to the IR: for every claim it records the realization candidates the optimizer
// weighed (each candidate's lane width + its scalarized cost), the one it chose, and the
// chosen edge's coupled score, plus any fusion credit (deforestation / CSE) that applied.
// The enclosing bcir.module carries `kbcir.explain_total` (the plan score). A reviewer (or
// `replay`) can read the record straight off the IR and re-derive why each claim got its
// width -- the same per-claim decision proof.explain emits, now first-class MLIR attributes:
//
//   per claim   kbcir.explain_chosen      : the chosen lane width
//               kbcir.explain_score       : the chosen edge cost (coupled, scalarized)
//               kbcir.explain_widths       : the candidate widths weighed (declared order)
//               kbcir.explain_candidates  : the parallel scalarized candidate costs
//               kbcir.explain_fusion       : "deforest" / "cse" when an intra-phase credit hit
//   per module  kbcir.explain_total       : the total plan score (== -bcir-plan's plan_score)
//
// -bcir-replay: the law-rail port of bcir/kbcir/proof.replay. A *recheck* of the record a
// deployed plan carries: it recomputes a fresh plan from the same inputs (the cost machinery
// of -bcir-plan / -bcir-explain) and DIFFS the fresh decision against the declared
// kbcir.explain_* record -- the module total and, per claim, the chosen width + edge score.
// Reproduction holds iff every field matches; the module gets `kbcir.replay_reproduced` (bool)
// and, when it diverged, `kbcir.replay_mismatches` (the per-field diff, mirroring
// ReplayResult.mismatches). The proof that a shipped plan is bit-for-bit re-derivable -- or
// exactly where it is not (a stale / tampered record).
//
// Read-only: the cost machinery is exactly -bcir-plan's (BCIRCostModel.h), so the chosen
// widths/score reproduce the oracle (7808 on vector_add) -- the record is the why, not a
// re-decision.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRCostModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <string>

using namespace mlir;

namespace bcir {
namespace {

// The fresh per-claim decision proof.replay diffs: the chosen lane width + the coupled edge
// score (proof.explain writes these as kbcir.explain_chosen / explain_score; replay rechecks
// them). The claim handle is a non-const copy so accessors are callable.
struct FreshDecision {
  ClaimOp claim;
  int64_t width;
  int64_t edge;
};

// Recompute the module's plan and the per-claim (width, edge score) + total -- the half of
// proof.explain's record proof.replay reproduces. Returns false if the module is not plannable.
static bool freshRecord(const cm::PlanAnalysis &pa, SmallVector<FreshDecision> &out,
                        int64_t &total) {
  if (!pa.valid)
    return false;
  const std::vector<cm::Column> &cols = pa.cols;
  const SmallVector<int> &chosen = pa.chosen;
  ArrayRef<int64_t> w = pa.weights;
  int64_t theta = pa.thetaThermal;
  total = pa.total;
  for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
    const cm::Column &col = cols[i];
    // The chosen edge cost: this candidate coupled by the chosen predecessor's context --
    // identical to -bcir-plan's / -bcir-explain's edge price.
    cm::Cost e = col.cands[chosen[i]].cost;
    cm::Factor f = (i > 0)
        ? cm::contextFactor(theta, cols[i - 1].reads,
                            cols[i - 1].cands[chosen[i - 1]].width, col.reads,
                            col.cands[chosen[i]].width)
        : cm::contextFactor(theta, {}, 0, col.reads, col.cands[chosen[i]].width);
    cm::applyFactor(e, f);
    FreshDecision d;
    d.claim = col.claim;
    d.width = col.cands[chosen[i]].width;
    d.edge = cm::scalarize(e, w);
    out.push_back(d);
  }
  return true;
}

struct ExplainPass : public PassWrapper<ExplainPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExplainPass)

  StringRef getArgument() const final { return "bcir-explain"; }
  StringRef getDescription() const final {
    return "Proof-carrying decision record (proof.explain) as IR annotations: per claim the "
           "candidates weighed (widths + scalarized costs), the chosen width + edge score, "
           "and any fusion credit; per module the total plan score (kbcir.explain_*).";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b, getChildAnalysis<cm::PlanAnalysis>(mod));
    });
    markAnalysesPreserved<cm::PlanAnalysis>();
  }

  void runOnModule(Operation *root, Builder &b, const cm::PlanAnalysis &pa) {
    if (!pa.valid)
      return;
    const std::vector<cm::Column> &cols = pa.cols;
    const SmallVector<int> &chosen = pa.chosen;
    ArrayRef<int64_t> w = pa.weights;
    int64_t theta = pa.thetaThermal;
    int64_t total = pa.total;

    for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
      const cm::Column &col = cols[i];
      // The candidates weighed: each candidate's width + its scalarized fused cost (the
      // path-INdependent half -- the menu the optimizer chose among, mirroring
      // proof.explain's per-candidate scores).
      SmallVector<int64_t> widths, scores;
      for (const cm::Cand &cand : col.cands) {
        widths.push_back(cand.width);
        scores.push_back(cm::scalarize(cand.cost, w));
      }
      // The chosen edge cost: this candidate coupled by the chosen predecessor's context
      // (source = no fusion predecessor; the thermal coupling still applies) -- identical
      // to -bcir-plan's edge price, so the record matches the plan it explains.
      cm::Cost e = col.cands[chosen[i]].cost;
      cm::Factor f = (i > 0)
          ? cm::contextFactor(theta, cols[i - 1].reads,
                              cols[i - 1].cands[chosen[i - 1]].width, col.reads,
                              col.cands[chosen[i]].width)
          : cm::contextFactor(theta, {}, 0, col.reads, col.cands[chosen[i]].width);
      cm::applyFactor(e, f);
      int64_t edge = cm::scalarize(e, w);

      col.claim->setAttr("kbcir.explain_chosen",
                         b.getI64IntegerAttr(col.cands[chosen[i]].width));
      col.claim->setAttr("kbcir.explain_score", b.getI64IntegerAttr(edge));
      col.claim->setAttr("kbcir.explain_widths", b.getDenseI64ArrayAttr(widths));
      col.claim->setAttr("kbcir.explain_candidates", b.getDenseI64ArrayAttr(scores));
      if (!col.fusion.empty())
        col.claim->setAttr("kbcir.explain_fusion", b.getStringAttr(col.fusion));
    }
    root->setAttr("kbcir.explain_total", b.getI64IntegerAttr(total));
  }
};

struct ReplayPass : public PassWrapper<ReplayPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplayPass)

  StringRef getArgument() const final { return "bcir-replay"; }
  StringRef getDescription() const final {
    return "Replay-check the proof-carrying record (proof.replay): recompute a fresh plan and "
           "diff it against the declared kbcir.explain_* record (module total + per-claim chosen "
           "width / edge score); annotates kbcir.replay_reproduced + replay_mismatches.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b, getChildAnalysis<cm::PlanAnalysis>(mod));
    });
    markAnalysesPreserved<cm::PlanAnalysis>();
  }

  void runOnModule(Operation *root, Builder &b, const cm::PlanAnalysis &pa) {
    // The recorded decision record the plan carries (what proof.explain emitted at plan time).
    // Nothing to replay if the module carries no record.
    auto recTotal = root->getAttrOfType<IntegerAttr>("kbcir.explain_total");
    if (!recTotal)
      return;

    SmallVector<FreshDecision> fresh;
    int64_t total = 0;
    if (!freshRecord(pa, fresh, total))
      return;

    // Diff the fresh plan against the record -- the module total, then each claim's chosen
    // width + edge score (proof.replay's per-claim (chosen, score, width) check).
    SmallVector<std::string> mismatches;
    if (total != recTotal.getInt())
      mismatches.push_back("total score " + std::to_string(total) + " != recorded " +
                           std::to_string(recTotal.getInt()));
    for (FreshDecision &d : fresh) {
      int64_t id = static_cast<int64_t>(d.claim.getClaimId());
      auto recWidth = d.claim->getAttrOfType<IntegerAttr>("kbcir.explain_chosen");
      auto recScore = d.claim->getAttrOfType<IntegerAttr>("kbcir.explain_score");
      if (!recWidth || !recScore) {
        mismatches.push_back("claim " + std::to_string(id) + " carries no recorded decision");
        continue;
      }
      if (d.width != recWidth.getInt() || d.edge != recScore.getInt())
        mismatches.push_back(
            "claim " + std::to_string(id) + ": replay (w" + std::to_string(d.width) + "/" +
            std::to_string(d.edge) + ") != recorded (w" + std::to_string(recWidth.getInt()) +
            "/" + std::to_string(recScore.getInt()) + ")");
    }

    root->setAttr("kbcir.replay_reproduced", b.getBoolAttr(mismatches.empty()));
    if (!mismatches.empty()) {
      SmallVector<Attribute> ms;
      for (const std::string &s : mismatches)
        ms.push_back(b.getStringAttr(s));
      root->setAttr("kbcir.replay_mismatches", b.getArrayAttr(ms));
    }
  }
};

} // namespace

std::unique_ptr<Pass> createExplainPass() { return std::make_unique<ExplainPass>(); }
std::unique_ptr<Pass> createReplayPass() { return std::make_unique<ReplayPass>(); }

} // namespace bcir
