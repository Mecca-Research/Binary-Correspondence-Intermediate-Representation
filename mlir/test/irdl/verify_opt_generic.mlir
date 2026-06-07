// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// M1 (verifier obligations as IR, R1-R12) + M2 (optimization law as IR) in
// generic syntax, round-tripped through the IRDL projection on stock mlir-opt.

"bcir.verify.registry_symbols"() {sym_name = "vr_symbols", registry = "RES",
  resources = ["A", "B", "C"], rids = [10, 11, 12]} : () -> ()
"bcir.verify.phase_dag"() {sym_name = "vr_phase", phases = ["p0"], edges = [],
  topological_order = [0]} : () -> ()
"bcir.verify.lane_stride"() {sym_name = "vr_lane", claim = "add", lane = "u",
  stride_class = "unit", stride_k = 1 : i32} : () -> ()
"bcir.verify.plan_selection"() {sym_name = "vr_plan", plan = "plan0", claim = "add",
  candidates = ["add_cpu_u8", "add_cpu_u16"], selected = "add_cpu_u16",
  semiring = "min_plus", score = 7808 : i64} : () -> ()
"bcir.verify.generation_tags"() {sym_name = "vr_gen", stream_pack = "sp0",
  topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 4 : i64,
  required_map_gen = 1 : i64, required_data_gen = 4 : i64} : () -> ()

"bcir.opt.pipeline"() {sym_name = "m2", stages = ["classify_lanes", "promote_stride",
  "score_kbcir", "select_min_plus_plan", "hydrate_gem_stream"]} : () -> ()
"bcir.opt.rewrite_rule"() {sym_name = "ggg_to_ux", kind = "lane_promotion",
  from_lane = "ggg", to_lane = "ux", from_stride = "random", to_stride = "cacheline",
  requires = "cacheline_bucket_proof", preserves = "claim_semantics,bounds,hazard",
  cost_axis = "memory"} : () -> ()
"bcir.opt.mem_rule"() {sym_name = "cxl_to_hbm", from_tier = "cxl", to_tier = "hbm",
  min_priority = 80 : i32, requires = "target_has_hbm,resource_hot", cost_axis = "memory"} : () -> ()

// CHECK: "bcir.verify.registry_symbols"
// CHECK: "bcir.verify.plan_selection"
// CHECK: "bcir.verify.generation_tags"
// CHECK: "bcir.opt.pipeline"
// CHECK: "bcir.opt.rewrite_rule"
// CHECK: "bcir.opt.mem_rule"
