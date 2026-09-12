// RUN: bcir-opt %s | bcir-opt | FileCheck %s
// The plan the root StreamPack was derived from travels as a BCAB variant of its own kind
// (ExecutionPlanV1, GEM+ G11): kind "execution_plan" requires format "execution_plan".

bcir.artifact.bundle @bundle attributes {
  version = 1 : i32, root_variant = "", default_variant = "01-plan",
  provenance_digest = "000000000000007b", generation = 7 : i64,
  wire_bytes = 648 : i64,
  artifact_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  body_crc32 = 1 : i64, header_crc32 = 2 : i64
} {
  bcir.artifact.variant @"01-plan" {
    kind = "execution_plan", format = "execution_plan", triple = "", architecture = "",
    os_abi = "", channel = "host", entry_symbol = "",
    endianness = "neutral", pointer_bits = 0 : i32, machine = 0 : i64,
    priority = 0 : i32, flags = 4 : i32,
    provenance_digest = "0000000000000000", required_features = [],
    prohibited_features = [], payload_offset = 576 : i64, payload_size = 68 : i64,
    payload_crc32 = 3 : i64,
    payload_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    target_manifest_sha256 = "", cal_gen = 0 : i64
  }
}

// CHECK: bcir.artifact.bundle @bundle
// CHECK: bcir.artifact.variant @"01-plan"
// CHECK: format = "execution_plan", kind = "execution_plan"
