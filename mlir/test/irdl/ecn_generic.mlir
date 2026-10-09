// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// G32: the ECN encoding objects (X.692, R24/R25) in generic syntax -- a module, a class, a
// structure and its fields, an object and its condition, a parameterized object --
// round-tripped through the IRDL projection. Region-bearing operations carry empty regions
// and their children are siblings (the structural rail's convention: an IRDL operation cannot
// declare NoTerminator).

"bcir.ecn_module"() ({}) {sym_name = "Frames", applies_to = @Frame} : () -> ()
"bcir.ecn_class"() {sym_name = "#Octets", base = "#BITS"} : () -> ()
"bcir.ecn_structure"() ({}) {sym_name = "Frame", encoding_class = @Octets} : () -> ()
"bcir.ecn_field"() {name = "length", encoding_class = @Octets, auxiliary} : () -> ()
"bcir.ecn_field"() {name = "value", encoding_class = @Octets} : () -> ()
"bcir.ecn_object"() ({}) {sym_name = "frame_enc", encoding_class = @Octets, align_unit = 8 : i64}
  : () -> ()
"bcir.ecn_condition"() {condition = "bounded", comparison = "lt", comparator = 256 : i64}
  : () -> ()
"bcir.ecn_parameterized"() {sym_name = "pad_to", kind = "object", dummies = ["n"],
  parameter_kinds = ["integer"], governors = ["n"]} : () -> ()

// CHECK: "bcir.ecn_module"
// CHECK: "bcir.ecn_class"
// CHECK: "bcir.ecn_structure"
// CHECK: "bcir.ecn_field"
// CHECK: "bcir.ecn_object"
// CHECK: "bcir.ecn_condition"
// CHECK: "bcir.ecn_parameterized"
