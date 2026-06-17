// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// IRDL smoke test in *generic* operation syntax (the pure-data form), loaded by
// stock mlir-opt with the BCIR IRDL projection registered at runtime -- no
// BCIR-authored C++. Validated on the latest LLVM/MLIR (22).
//
// The IRDL rail is structural: region-bearing ops are exercised with empty
// regions (the nested tree is carried by the ODS rail). Leaf ops are siblings.

"bcir.module"() ({}) {
  sym_name = "smoke", version = "0.1", target_model = "registry-first",
  execution_model = "gem", cost_model = "k_bcir"
} : () -> ()
"bcir.registry"() ({}) {sym_name = "RES"} : () -> ()
%A = "bcir.resource"() {
  sym_name = "A", rid = 10 : i32, domain_kind = "ram", shape = [1024],
  layout = "soa", align = 64 : i32, access = "flat", priority = 0 : i32,
  map_gen = 1 : i64, data_gen = 4 : i64
} : () -> !bcir.handle
"bcir.phase"() {sym_name = "p0", id = 0 : i32, deps = []} : () -> ()
"bcir.target_capability"() {sym_name = "cpu", triple = "x86_64-avx512",
  isa_features = ["avx2", "avx512f", "fma"], lane_widths = [1, 8, 16],
  warp = 0 : i32, cacheline = 64 : i32} : () -> ()
"bcir.mem_tier"() {sym_name = "hbm", tier = "hbm", latency_cyc = 160 : i64,
  bw_factor = 16384 : i64, lat_factor = 49152 : i64} : () -> ()

// CHECK: "bcir.module"
// CHECK: "bcir.registry"
// CHECK: "bcir.resource"
// CHECK: !bcir.handle
// CHECK: "bcir.target_capability"
// CHECK: "bcir.mem_tier"
