// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// BCIR-3 (K_BCIR) + BCIR-4 (GEM) in generic syntax, round-tripped through the
// IRDL projection on stock mlir-opt. Mirrors the oracle's plan + StreamPack
// (see docs/PARITY.md): vec16 selected with score 7808.

"bcir.kbcir.policy"() {sym_name = "perf", mode = "latency",
  weights = [2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1]} : () -> ()
%p8 = "bcir.kbcir.path"() {sym_name = "add_cpu_u8", claim = "add",
  realization = "cpu.vector.u8", lane = "u", layout = "soa"} : () -> !bcir.handle
%p16 = "bcir.kbcir.path"() {sym_name = "add_cpu_u16", claim = "add",
  realization = "cpu.vector.u16", lane = "u", layout = "soa"} : () -> !bcir.handle
%sel = "bcir.kbcir.select"() {claim = "add", from = ["add_cpu_u8", "add_cpu_u16"],
  policy = "latency", semiring = "min_plus", selected = "add_cpu_u16", score = 7808 : i64} : () -> !bcir.handle
"bcir.kbcir.plan"() ({}) {sym_name = "plan0"} : () -> ()
%sp = "bcir.gem.stream_pack"() ({}) {sym_name = "sp0", source_plan = "plan0",
  topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 4 : i64} : () -> !bcir.handle
"bcir.gem.prefetch"() {sym_name = "pf0", distance = 4 : i32, targets = ["A", "B"],
  hint = "T0", pattern = "linear"} : () -> ()
"bcir.gem.lane_segment"() {sym_name = "seg0", claim = "add", phase = "p0", lane = "u",
  stride_k = 1 : i32, width = 16 : i32, opcode = "f32.add", reads = ["A", "B"],
  writes = ["C"], fence_before = [], fence_after = []} : () -> ()
"bcir.trace.note"() {sym_name = "add_trace", src_hash = 0 : i64, trace_hash = 0 : i64} : () -> ()

// CHECK: "bcir.kbcir.policy"
// CHECK: "bcir.kbcir.path"
// CHECK: "bcir.kbcir.select"
// CHECK: "bcir.gem.stream_pack"
// CHECK: "bcir.gem.lane_segment"
// CHECK: "bcir.trace.note"
