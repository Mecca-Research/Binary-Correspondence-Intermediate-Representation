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

def BCIR_KBCIRThetaOp : BCIR_Op<"kbcir.theta", [Symbol]> {
  let summary = "Live runtime state Theta (0..100 normalized pressures)";
  // The runtime state the K_BCIR equation conditions on (mirrors bcir/kbcir/cost.py
  // Theta). The emitter folds Theta into the policy weights for scalarization; this
  // op additionally carries the raw pressures the *multiplicative* context coupling
  // needs -- -bcir-plan / -bcir-overlap read `thermal` to apply the AVX-512 downclock
  // (a wide width>=16 candidate on a hot part, thermal>=60, pays x1.25 heat/current).
  // Absent (or thermal 0) == the cool regime. Defaulted so existing IR parses.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       DefaultValuedAttr<I32Attr, "0">:$thermal,
                       DefaultValuedAttr<I32Attr, "0">:$power,
                       DefaultValuedAttr<I32Attr, "0">:$mem_pressure,
                       DefaultValuedAttr<I32Attr, "0">:$contention);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRPlanOp : BCIR_Op<"kbcir.plan", [Symbol, SymbolTable, NoTerminator]> {
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
  // A Bayesian/conformal table (kbcir.bayescal) additionally carries
  // `coverage_milli` (the split-conformal coverage x1000; 0 = a plain point
  // table, no guarantee) and `random_delta_q8` (the certified +/- half-width on
  // the random ratio). R8: coverage in (0,1000) and delta >= 0 when claimed.
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$target,
                       I64Attr:$cal_gen, I64Attr:$samples, StrAttr:$provenance,
                       I64Attr:$stream_q8, I64Attr:$strided_q8,
                       I64Attr:$random_q8, I64Attr:$compute_q8,
                       DefaultValuedAttr<I64Attr, "0">:$coverage_milli,
                       DefaultValuedAttr<I64Attr, "0">:$random_delta_q8);
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
  // The retune trigger is the MDL / Bayesian-evidence two-part code, not a
  // magic threshold: `data_fit_milli` (sum_i regret_i/best_i, in milli-nats) is
  // the description-length saving a swap buys; `complexity_milli` is the BIC
  // penalty (k/2)*ln(N) it costs. `verdict` is "keep" or "retune", and R13
  // checks it is consistent: retune <=> data_fit_milli > complexity_milli.
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$rule,
                       I64Attr:$episodes, I64Attr:$total_regret,
                       I64Attr:$worst_regret, I64Attr:$gen,
                       DefaultValuedAttr<I64Attr, "0">:$data_fit_milli,
                       DefaultValuedAttr<I64Attr, "0">:$complexity_milli,
                       DefaultValuedAttr<StrAttr, "\"keep\"">:$verdict);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRSoftSelectOp : BCIR_Op<"kbcir.soft_select", [Symbol]> {
  let summary = "Finite-temperature (log-sum-exp) twin of the tropical select";
  // The Gibbs free energy F_T = -T log sum exp(-score/T) over legal plans -- the
  // soft, differentiable generalization of kbcir.select (LangRef Sec. 8).
  // `temperature_milli` is T x 1000 (integer for IR determinism). Degenerate-case
  // law (verified, R9): free_energy <= score (softmin <= min) and, at
  // temperature 0, free_energy == score -- exact recovery of the tropical
  // optimizer. Mirrors bcir/kbcir/softdp.py.
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$claim,
                       I64Attr:$temperature_milli, I64Attr:$score,
                       I64Attr:$free_energy, FlatSymbolRefAttr:$selected);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRMoEGateOp : BCIR_Op<"kbcir.moe_gate", [Symbol]> {
  let summary = "L2 learned MoE gate: a GNN router over the claim graph (the ensemble)";
  // The learned replacement for classify(Theta) (LangRef Sec. 13): a GNN over the
  // claim graph routes among the certified `portfolio` experts. It is the *safe*
  // learning regime -- it selects among already-certified experts, never emits a
  // table -- and deploys frozen (Q8 integer routing) only behind an admitting
  // replay certificate. `fingerprint` is the frozen params' provenance hash.
  // R13: dims positive, portfolio resolves, admitted => regressions == 0, and a
  // deployed gate must be admitted. Mirrors bcir/kbcir/moegate.py.
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$portfolio,
                       I64Attr:$num_experts, I64Attr:$hidden, I64Attr:$gate_gen,
                       I64Attr:$fingerprint, I64Attr:$episodes, I64Attr:$regressions,
                       BoolAttr:$admitted);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRSearchAccelOp : BCIR_Op<"kbcir.search_accel", [Symbol]> {
  let summary = "Propose-verify search accelerator: a learned candidate ordering";
  // A learned order over the RCSP candidate DAG speeds the exact branch-and-bound
  // search but must reproduce the exact optimum (LangRef Sec. 13): ordering
  // changes work, never the result. `order` names the heuristic (greedy/learned);
  // `mismatches` counts cases where the accelerated optimum differed from the
  // exact one over `checked` cases. R13: admitted => mismatches == 0, and a
  // deployed accelerator must be admitted. Mirrors bcir/kbcir/accel.py.
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$order,
                       I64Attr:$checked, I64Attr:$mismatches, BoolAttr:$admitted);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRProvenanceManifestOp : BCIR_Op<"kbcir.provenance_manifest", [Symbol]> {
  let summary = "Provenance manifest: the commit hash of a plan (R13 reproducibility)";
  // The content hash chaining a plan's inputs (module/target/theta/policy) and the
  // in-force decision-rule generations, with the recorded optimal `score`
  // (LangRef Sec. 13). Manifest equality => identical plan; an immutable plan is a
  // committed manifest. R13: a deployed plan's manifest must have `reproduced` ==
  // true (replay reproduced the recorded score/shape). Mirrors
  // bcir/kbcir/provenance.py + verify.verify_manifest.
  let arguments = (ins SymbolNameAttr:$sym_name, I64Attr:$digest, I64Attr:$score,
                       I64Attr:$n_artifacts, BoolAttr:$reproduced);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRAmortizationOp : BCIR_Op<"kbcir.amortization", [Symbol]> {
  let summary = "L1 cost throttle: a learned component's tier-placement certificate";
  // Keeps performance throttled where it matters (LangRef Sec. 13): a deployed
  // learned component declares its plan-time `inference_cost`, the `gain` it
  // buys, and the tier `budget`. R13: tier "L0" => inference_cost == 0 (the hot
  // path carries decisions, not models); otherwise gain >= inference_cost (it
  // pays for itself) and inference_cost <= budget (the throttle). Mirrors
  // bcir/kbcir/throttle.py.
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$component, StrAttr:$tier,
                       I64Attr:$inference_cost, I64Attr:$gain, I64Attr:$budget);
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

// --- compositional semantics: the func / if op family (BCIR-3 compose) -----------
// The IR vocabulary for the region tree of bcir/kbcir/compose.py -- the frontier past
// straight-line array kernels. A `kbcir.func` is a named region (define once, call many);
// `kbcir.call` invokes it, substituting actual resources for the callee's formals; and
// `kbcir.cond` is control flow (worst-case max + probability-weighted expected). The
// compositional cost (series sum / branch max / call inline-or-summary) is computed by the
// oracle, which stays the conformance reference; these ops give that program structure a
// first-class law-rail form (annotated/analysed incrementally, like the kbcir.* family above).

def BCIR_KBCIRFuncOp : BCIR_Op<"kbcir.func", [Symbol, SymbolTable, NoTerminator]> {
  let summary = "A named compositional region (compose.Function): define once, kbcir.call many";
  // `formals` names the callee's formal resources (the symbols a kbcir.call remaps); the
  // body region holds the function's claims / nested calls / conds.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<FlatSymbolRefArrayAttr>:$formals);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_KBCIRCallOp : BCIR_Op<"kbcir.call"> {
  let summary = "Invoke a kbcir.func, substituting actual resources for its formals (compose.Call)";
  // `callee` is the kbcir.func; `formals`/`actuals` are the parallel substitution arrays
  // (formal_i -> actual_i). Inline-substitution semantics: the planner sees the expanded
  // graph (or, with an inter-procedural summary, reuses the callee's cost). Recursion is
  // rejected by the oracle (bounded compile time).
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       OptionalAttr<FlatSymbolRefArrayAttr>:$formals,
                       OptionalAttr<FlatSymbolRefArrayAttr>:$actuals);
  let assemblyFormat = "$callee attr-dict";
}

def BCIR_KBCIRCondOp : BCIR_Op<"kbcir.cond", [NoTerminator]> {
  let summary = "Control-flow region (compose.Cond): worst-case max + prob-weighted expected";
  // Only one branch runs, so the hard latency bound is the max over the two branches and the
  // expected cost weights by `prob_then_milli`/1000 (defaulted 500 = even odds). `pred` names
  // the predicate the branch keys on.
  let arguments = (ins StrAttr:$pred,
                       DefaultValuedAttr<I64Attr, "500">:$prob_then_milli);
  let regions = (region SizedRegion<1>:$then_region, SizedRegion<1>:$else_region);
  let assemblyFormat =
      "$pred attr-dict-with-keyword `then` $then_region `else` $else_region";
}

#endif // BCIR_KBCIR_OPS
