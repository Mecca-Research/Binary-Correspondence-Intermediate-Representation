// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// CT2 (concurrency/affinity: a lane segment pinned to an affinity domain with a
// cache-unroll factor) + CT4 (a data-DNA telemetry record) in generic syntax,
// round-tripped through the IRDL projection on stock mlir-opt.

"bcir.gem.lane_segment"() {sym_name = "seg0", claim = "add", phase = "p0", lane = "u",
  stride_k = 1 : i32, width = 16 : i32, opcode = "f32.add", reads = ["A", "B"],
  writes = ["C"], fence_before = [], fence_after = [],
  affinity = 3 : i32, unroll = 4 : i32} : () -> ()

"bcir.trace.data_dna"() {sym_name = "dna0", segment = "seg0", cycles = 12000 : i64,
  bytes = 49152 : i64, misses = 12 : i32, thermal = 80 : i32, voltage = 10 : i32,
  utilization = 70 : i32, provenance = "plan0:add"} : () -> ()

// CHECK: "bcir.gem.lane_segment"
// CHECK: affinity = 3
// CHECK: unroll = 4
// CHECK: "bcir.trace.data_dna"
// CHECK: thermal = 80
