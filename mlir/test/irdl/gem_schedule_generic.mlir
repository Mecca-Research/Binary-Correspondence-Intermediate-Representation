// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// Duration-aware GEM scheduling + StreamPack v2 pipelining in generic syntax:
// an EFT schedule certificate (LPT + earliest finish + locality + the
// bandwidth knee), a token-DAG schedule, a double-buffer prefetch contract
// (buffers=2), and a pipelined stream pack (pipeline_depth=2). Mirrors
// bcir/gem/schedule.py + hydrate_pipelined (docs/PARITY.md).

"bcir.kbcir_plan"() ({}) {sym_name = "plan0"} : () -> ()
"bcir.gem_schedule"() {sym_name = "sched_eft", plan = "plan0", mode = "eft",
  makespan = 11 : i64, knee = 2 : i32} : () -> ()
"bcir.gem_schedule"() {sym_name = "sched_tokens", plan = "plan0", mode = "tokens",
  makespan = 8 : i64, knee = 2 : i32, pipeline_depth = 2 : i32} : () -> ()
"bcir.gem_prefetch"() {sym_name = "dbpf0_1", distance = 4 : i32, targets = ["B"],
  hint = "T1", pattern = "double_buffer", buffers = 2 : i32} : () -> ()
"bcir.gem_lane_segment"() {sym_name = "seg0", claim = "add", phase = "p0",
  lane = "u", stride_k = 1 : i32, width = 16 : i32, opcode = "f32.add",
  reads = ["A", "B"], writes = ["C"], fence_before = [], fence_after = [],
  duration = 7808 : i64} : () -> ()
%sp = "bcir.gem_stream_pack"() ({}) {sym_name = "sp0", source_plan = "plan0",
  topo_gen = 1 : i64, map_gen = 0 : i64, data_gen = 0 : i64,
  pipeline_depth = 2 : i32} : () -> !bcir.handle

// CHECK: "bcir.gem_schedule"
// CHECK: mode = "eft"
// CHECK: mode = "tokens"
// CHECK: pattern = "double_buffer"
// CHECK: pipeline_depth = 2
