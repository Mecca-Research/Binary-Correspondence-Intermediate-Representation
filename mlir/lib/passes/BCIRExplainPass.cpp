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

using namespace mlir;

namespace bcir {
namespace {

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
        runOnModule(mod, b);
    });
  }

  void runOnModule(Operation *root, Builder &b) {
    auto capOp = cm::firstCapability(root);
    if (!capOp)
      return;
    cm::Cap h = cm::readCap(capOp);
    ArrayRef<int64_t> w = cm::firstWeights(root);
    if (w.empty())
      return;
    auto resByName = cm::resourcesByName(root);
    std::vector<cm::Column> cols = cm::fusedColumns(root, h, resByName);
    if (cols.empty())
      return;

    int64_t theta = cm::firstThetaThermal(root);
    int64_t total = 0;
    SmallVector<int> chosen = cm::planChosen(cols, w, theta, total);
    if (chosen.empty())
      return;

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

} // namespace

std::unique_ptr<Pass> createExplainPass() { return std::make_unique<ExplainPass>(); }

} // namespace bcir
