//===- BCIRKBCIROps.td - K_BCIR planning ops (BCIR-3) -------*- tablegen -*-===//
#ifndef BCIR_KBCIR_OPS
#define BCIR_KBCIR_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRTypes.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_KBCIRPolicyOp : BCIR_Op<"kbcir.policy", [Symbol]> {
  let summary = "K_BCIR scalarization policy (12 Q16 weights)";
  let arguments = (ins SymbolNameAttr:$sym_name, BCIR_PolicyModeAttr:$mode,
                       DenseI64ArrayAttr:$weights);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRPlanOp : BCIR_Op<"kbcir.plan", [Symbol, NoTerminator]> {
  let summary = "K_BCIR candidate-path + selected-plan region";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_KBCIRPathOp : BCIR_Op<"kbcir.path", [Symbol]> {
  let summary = "A feasible realization path for a BCIR claim";
  let arguments = (ins
    SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$claim, StrAttr:$realization,
    BCIR_LaneAttr:$lane, BCIR_LayoutAttr:$layout, BCIR_CostVectorAttr:$cost);
  let results = (outs BCIR_PathType:$path);
  let assemblyFormat = "$sym_name attr-dict `:` type($path)";
}

def BCIR_KBCIRSelectOp : BCIR_Op<"kbcir.select"> {
  let summary = "Select a realization path under a K_BCIR policy (min-plus)";
  // An optional `budget` switches the selection to the constrained (RCSP) rail:
  // the selected path must satisfy the referenced caps (verified, law R9). No
  // budget = the scalarized degenerate case (LangRef Sec. 2).
  let arguments = (ins
    FlatSymbolRefAttr:$claim, FlatSymbolRefArrayAttr:$from,
    BCIR_PolicyModeAttr:$policy, BCIR_SemiringAttr:$semiring,
    FlatSymbolRefAttr:$selected, I64Attr:$score,
    OptionalAttr<FlatSymbolRefAttr>:$budget);
  let results = (outs BCIR_PathType:$result);
  let assemblyFormat = "$claim `from` $from attr-dict `:` type($result)";
}

def BCIR_KBCIRBudgetOp : BCIR_Op<"kbcir.budget", [Symbol]> {
  let summary = "B(H,Theta): hard caps on additive resource dims for RCSP selection";
  // `dims` names cost-vector dimensions (compute/memory/.../verification);
  // `caps` are the parallel integer caps. Mirrors bcir/kbcir/rcsp.Budget.
  let arguments = (ins SymbolNameAttr:$sym_name, StrArrayAttr:$dims,
                       DenseI64ArrayAttr:$caps);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRCalibrationOp : BCIR_Op<"kbcir.calibration", [Symbol]> {
  let summary = "L1 frozen cost table: measured Q8 ratios, generation-tagged";
  // The physics-anchored calibration certificate (LangRef Sec. 13): host
  // microbenchmark ratios against the streaming baseline (stream_q8 == 256 by
  // definition, verified under R8). gather_penalty = random_q8/256,
  // base_overhead = 4*strided_q8/256. Mirrors bcir/kbcir/microbench.py.
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$target,
                       I64Attr:$cal_gen, I64Attr:$samples, StrAttr:$provenance,
                       I64Attr:$stream_q8, I64Attr:$strided_q8,
                       I64Attr:$random_q8, I64Attr:$compute_q8);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRPortfolioOp : BCIR_Op<"kbcir.portfolio", [Symbol]> {
  let summary = "L2 policy portfolio: frozen gain schedules + certification state";
  // `policies` reference kbcir.policy ops; `gens`/`certified` are parallel
  // arrays (certified: 0/1). Plan-time selection is a workload-class table
  // lookup; mutation only through an admitting replay certificate. Mirrors
  // bcir/kbcir/portfolio.py (verified under R9).
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefArrayAttr:$policies,
                       DenseI64ArrayAttr:$gens, DenseI64ArrayAttr:$certified);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRReplayCertificateOp : BCIR_Op<"kbcir.replay_certificate", [Symbol]> {
  let summary = "L2 replay gate: counterfactual no-regression policy-swap certificate";
  // admitted requires regressions == 0 over >= 1 episodes (verified, R9):
  // a gain schedule never deploys on a measured loss.
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$candidate,
                       FlatSymbolRefAttr:$incumbent, I64Attr:$episodes,
                       I64Attr:$regressions, BoolAttr:$admitted);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRRegretLedgerOp : BCIR_Op<"kbcir.regret_ledger", [Symbol]> {
  let summary = "L3 regret ledger: a deployed rule's measured gap to hindsight";
  // The boundary dashboard (LangRef Sec. 13): per-rule accumulated regret
  // against the hindsight-best alternative under one neutral yardstick.
  // An instrument, never an actuator: verdicts recommend, the replay gate
  // decides, R13 witnesses. Mirrors bcir/kbcir/regret.py (verified, R13).
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$rule,
                       I64Attr:$episodes, I64Attr:$total_regret,
                       I64Attr:$worst_regret, I64Attr:$gen);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRScheduledPriceOp : BCIR_Op<"kbcir.scheduled_price", [Symbol]> {
  let summary = "M(pi,Theta): the (max,+) wave-overlap price of a plan";
  // makespan composes series with (+) and CT2 wave parallelism with max; the
  // serial Sigma score is the degenerate bound: makespan + overlap_gain ==
  // serial (verified, law R9). Mirrors bcir/gem/overlap.price_scheduled.
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$plan,
                       I64Attr:$makespan, I64Attr:$serial, I64Attr:$overlap_gain);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_KBCIR_OPS
