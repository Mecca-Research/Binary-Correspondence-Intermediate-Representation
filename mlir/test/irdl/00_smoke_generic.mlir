// RUN: mlir-opt <LOAD_BCIR_IRDL_FLAGS> %s | FileCheck %s
//
// IRDL smoke test in *generic* operation syntax (the pure-data form). The
// concrete load flag is whatever `tools/irdl/probe_irdl_flags.sh` reports on the
// installed mlir-opt (e.g. --irdl-file=mlir/irdl/bcir.irdl.mlir). Not run on this
// host (no mlir-opt).

"bcir.module"() ({
  "bcir.registry"() ({
    %A = "bcir.resource"() {
      sym_name = "A", rid = 10 : i32, domain_kind = "ram", shape = [1024],
      layout = "soa", align = 64 : i32, access = "flat", priority = 0 : i32,
      map_gen = 1 : i64, data_gen = 4 : i64
    } : () -> !bcir.resource
  }) {sym_name = "RES"} : () -> ()

  "bcir.phase"() {sym_name = "p0", id = 0 : i32, deps = []} : () -> ()
}) {
  sym_name = "smoke", version = "0.1", target_model = "registry-first",
  execution_model = "gem", cost_model = "k_bcir"
} : () -> ()

// CHECK: "bcir.module"
// CHECK: "bcir.registry"
// CHECK: "bcir.resource"
// CHECK: !bcir.resource
