// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
// BCAB has a loose pure-data IRDL projection in addition to the strict ODS rail.

"bcir.artifact_bundle"() ({}) {sym_name = "bundle", version = 1 : i32} : () -> ()
"bcir.artifact_variant"() {sym_name = "portable-c", kind = "c_source", format = "text",
  payload_size = 12 : i64, payload_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"} : () -> ()
"bcir.artifact_selection"() {sym_name = "selection", bundle = @bundle,
  variant = "portable-c", classification = "exact"} : () -> ()

// CHECK: "bcir.artifact_bundle"
// CHECK: "bcir.artifact_variant"
// CHECK: "bcir.artifact_selection"
