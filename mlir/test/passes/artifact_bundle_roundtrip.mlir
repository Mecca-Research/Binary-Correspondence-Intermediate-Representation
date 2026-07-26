// RUN: bcir-opt %s | bcir-opt | FileCheck %s

bcir.artifact.bundle @bundle attributes {
  version = 1 : i32, root_variant = "", default_variant = "portable-c",
  provenance_digest = "000000000000007b", generation = 7 : i64,
  wire_bytes = 584 : i64,
  artifact_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  body_crc32 = 1 : i64, header_crc32 = 2 : i64
} {
  bcir.artifact.variant @"portable-c" {
    kind = "c_source", format = "text", triple = "", architecture = "",
    os_abi = "", channel = "", entry_symbol = "bcir_kernel",
    endianness = "neutral", pointer_bits = 0 : i32, machine = 0 : i64,
    priority = 0 : i32, flags = 4 : i32,
    provenance_digest = "0000000000000000", required_features = [],
    prohibited_features = [], payload_offset = 576 : i64, payload_size = 4 : i64,
    payload_crc32 = 3 : i64,
    payload_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    target_manifest_sha256 = "", cal_gen = 0 : i64
  }
}

bcir.artifact.selection @chosen {
  bundle = @bundle, variant = "portable-c", classification = "exact",
  envelope_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  generation = 7 : i64
}

// CHECK: bcir.artifact.bundle @bundle
// CHECK: bcir.artifact.variant @"portable-c"
// CHECK: kind = "c_source"
// CHECK: bcir.artifact.selection @chosen
// CHECK: classification = "exact"
