// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// The constrained series-parallel rail (LangRef Sec. 2) in generic syntax:
// B(H,Theta) budget caps, a budgeted kbcir.select, and the (max,+) scheduled
// price M(pi,Theta). Mirrors the oracle (docs/PARITY.md): a 700 thermal cap
// makes vec16 infeasible and selects vec8 at score 9472; a single-claim plan
// prices at makespan == serial 7808 (the degenerate case).

"bcir.kbcir.budget"() {sym_name = "thermal_cap", dims = ["thermal", "power"],
  caps = array<i64: 700, 700>} : () -> ()
%p8 = "bcir.kbcir.path"() {sym_name = "add_cpu_u8", claim = "add",
  realization = "cpu.vector.u8", lane = "u", layout = "soa"} : () -> !bcir.handle
%sel = "bcir.kbcir.select"() {claim = "add", from = ["add_cpu_u8"],
  policy = "latency", semiring = "min_plus", budget = "thermal_cap",
  selected = "add_cpu_u8", score = 9472 : i64} : () -> !bcir.handle
"bcir.kbcir.plan"() ({}) {sym_name = "plan0"} : () -> ()
"bcir.kbcir.scheduled_price"() {sym_name = "overlap_price", plan = "plan0",
  makespan = 7808 : i64, serial = 7808 : i64, overlap_gain = 0 : i64} : () -> ()

// CHECK: "bcir.kbcir.budget"
// CHECK: "bcir.kbcir.select"
// CHECK: "bcir.kbcir.scheduled_price"
