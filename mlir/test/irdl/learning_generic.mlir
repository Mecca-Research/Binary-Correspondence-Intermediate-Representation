// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// Learning placement (LangRef Sec. 13) in generic syntax: the L1 frozen
// calibration table (ratio-1 reference reproduces the seeded constants:
// gather_penalty 32, base_overhead 4), the L2 certified policy portfolio, and
// the replay gate certificate. Mirrors bcir/kbcir/{microbench,portfolio}.py
// (docs/PARITY.md).

"bcir.target.capability"() {sym_name = "cpu", triple = "x86_64-avx512",
  isa_features = ["avx512f"], lane_widths = [1, 8, 16], cal_gen = 1 : i64} : () -> ()
"bcir.kbcir.calibration"() {sym_name = "cal_cpu", target = "cpu",
  cal_gen = 1 : i64, samples = 0 : i64,
  provenance = "reference (ratio-1; reproduces the seeded constants exactly)",
  stream_q8 = 256 : i64, strided_q8 = 256 : i64, random_q8 = 8192 : i64,
  compute_q8 = 256 : i64} : () -> ()
"bcir.kbcir.policy"() {sym_name = "latency", mode = "latency",
  weights = [2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1]} : () -> ()
"bcir.kbcir.portfolio"() {sym_name = "gains", policies = ["latency"],
  gens = array<i64: 1>, certified = array<i64: 1>} : () -> ()
"bcir.kbcir.replay_certificate"() {sym_name = "cert", candidate = "latency",
  incumbent = "latency", episodes = 3 : i64, regressions = 0 : i64,
  admitted = true} : () -> ()
"bcir.kbcir.regret_ledger"() {sym_name = "regret_latency", rule = "latency",
  episodes = 3 : i64, total_regret = 0 : i64, worst_regret = 0 : i64,
  gen = 1 : i64} : () -> ()
"bcir.verify.policy_provenance"() {sym_name = "vr_policy", portfolio = "gains",
  certificates = ["cert"], calibrations = ["cal_cpu"]} : () -> ()
"bcir.kbcir.moe_gate"() {sym_name = "gate0", portfolio = "gains", num_experts = 4 : i64,
  hidden = 8 : i64, gate_gen = 1 : i64, fingerprint = 4242424242 : i64,
  episodes = 3 : i64, regressions = 0 : i64, admitted = true} : () -> ()
"bcir.kbcir.search_accel"() {sym_name = "accel0", order = "learned",
  checked = 4 : i64, mismatches = 0 : i64, admitted = true} : () -> ()
"bcir.kbcir.provenance_manifest"() {sym_name = "manifest0", digest = 7777777 : i64,
  score = 7808 : i64, n_artifacts = 0 : i64, reproduced = true} : () -> ()
"bcir.egraph.extract"() {sym_name = "egraph0", original_cost = 5 : i64,
  optimized_cost = 3 : i64, iterations = 3 : i64, enodes = 11 : i64, saturated = true} : () -> ()
"bcir.kbcir.amortization"() {sym_name = "throttle_gate", component = "moe_gate",
  tier = "L1", inference_cost = 300 : i64, gain = 6000 : i64, budget = 100000 : i64} : () -> ()

// CHECK: "bcir.kbcir.calibration"
// CHECK: random_q8 = 8192
// CHECK: "bcir.kbcir.portfolio"
// CHECK: "bcir.kbcir.replay_certificate"
// CHECK: admitted = true
// CHECK: "bcir.kbcir.regret_ledger"
// CHECK: "bcir.verify.policy_provenance"
// CHECK: "bcir.kbcir.moe_gate"
// CHECK: "bcir.kbcir.search_accel"
// CHECK: "bcir.kbcir.provenance_manifest"
// CHECK: "bcir.egraph.extract"
// CHECK: "bcir.kbcir.amortization"
